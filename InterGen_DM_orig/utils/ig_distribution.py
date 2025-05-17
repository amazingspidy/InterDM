import numpy as np
import torch as th
import torch.nn.functional as F
import torchgeometry as tgm


def align_points(points1, points2):
    """
    Aligns two 3D point clouds using the Kabsch algorithm.
    
    Args:
        points1: (B, N, 3) tensor, source point cloud
        points2: (B, N, 3) tensor, target point cloud
    
    Returns:
        rotation: (B, 3, 3) tensor, optimal rotation matrices
        translation: (B, 3) tensor, optimal translation vectors
    """
    B, N, _ = points1.shape
    
    # Compute centroids
    centroid1 = points1.mean(dim=1, keepdim=True)  # (B, 1, 3)
    centroid2 = points2.mean(dim=1, keepdim=True)  # (B, 1, 3)
    
    # Center the points
    points1_centered = points1 - centroid1  # (B, N, 3)
    points2_centered = points2 - centroid2  # (B, N, 3)
    
    # Compute covariance matrix
    H = th.bmm(points1_centered.transpose(1, 2), points2_centered)  # (B, 3, 3)
    
    # Compute SVD
    U, S, Vt = th.svd(H)
    
    # Compute rotation
    R = th.bmm(Vt, U.transpose(1, 2))
    
    # Ensure a right-handed coordinate system
    det_R = th.det(R)
    Vt[:, :, -1] *= th.sign(det_R).unsqueeze(-1)
    R = th.bmm(Vt, U.transpose(1, 2))
    
    # Compute translation
    t = centroid2.squeeze(1) - th.bmm(R, centroid1.squeeze(1).unsqueeze(-1)).squeeze(-1)  # (B, 3)
    
    return R, t

def axis_angle_to_rotation_matrix(angle_axis: th.Tensor) -> th.Tensor:
    r"""
    Converts an axis-angle vector (shape: (..., 3)) into a rotation matrix (shape: (..., 3, 3))
    using Rodrigues’ formula:
    
      R = I + A [r]_x + B [r]_x²,
    
    where r = θ·u with θ = ||r||, and
      A = sinθ/θ,   B = (1-cosθ)/θ².
    
    To avoid 0/0 issues (and nan in the backward pass) when r is near 0,
    we replace θ by sqrt(θ²+ε) (with a small ε) in denominators.
    
    This ensures that when angle_axis == 0, we get exactly R == I and a finite gradient.
    """
    eps = 1e-12  # 아주 작은 상수
    # squared norm: θ² = r_x² + r_y² + r_z², shape (..., 1)
    theta_sq = th.sum(angle_axis ** 2, dim=-1, keepdim=True)
    # 안전하게 sqrt를 계산: θ = sqrt(θ² + eps)
    theta = th.sqrt(theta_sq + eps)

    # 안전한 분모 사용: r이 0일 때도 분모가 0이 되지 않음.
    A = th.sin(theta) / theta          # ≈ 1 when θ ~ 0
    B = (1 - th.cos(theta)) / (theta_sq + eps)  # ≈ 0.5 when θ ~ 0

    # skew-symmetric matrix [r]_x
    # 각 성분을 (..., 1) 모양으로 만들어 shape mismatch를 피합니다.
    zero = th.zeros_like(theta)
    ax0 = angle_axis[..., 0:1]
    ax1 = angle_axis[..., 1:2]
    ax2 = angle_axis[..., 2:3]
    K = th.cat([
        zero,    -ax2,   ax1,
        ax2,     zero,   -ax0,
       -ax1,     ax0,    zero
    ], dim=-1).view(angle_axis.shape[:-1] + (3, 3))
    
    # 3x3 identity matrix, 배치에 맞게 확장
    I = th.eye(3, device=angle_axis.device, dtype=angle_axis.dtype)
    I = I.expand(angle_axis.shape[:-1] + (3, 3))
    
    # Rodrigues’ formula 적용
    # A와 B는 (..., 1) 모양이므로, unsqueeze(-1)를 통해 (..., 1, 1)로 만듭니다.
    R = I + A.unsqueeze(-1) * K + B.unsqueeze(-1) * (K @ K)
    return R

def rotation_matrix_to_axis_angle(R: th.Tensor) -> th.Tensor:
    r"""
    Rotation matrix (..., 3, 3) 를 axis-angle (..., 3) (즉, log map)으로 변환합니다.
    
    일반적으로 아래 공식이 사용됩니다.
      r = \frac{\theta}{2\sin\theta}(R - R^T)^\vee,   with   cosθ = (trace(R)-1)/2.
    
    다만, θ≈0인 경우 0/0 문제가 발생하므로, 
    Taylor 전개에 따른 작은 각 근사를 smooth하게 보간합니다.
    
    입력:
      R: (..., 3, 3) 회전행렬
    출력:
      angle_axis: (..., 3) axis-angle (r = θ·axis)
    """
    # vee 연산: (R - R^T)에서 3-vector 추출
    r_vec = th.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1)  # (..., 3)

    # norm를 안정적으로 계산 (작은 eps를 더함)
    eps_norm = 1e-12
    norm_r = th.sqrt(th.sum(r_vec**2, dim=-1, keepdim=True) + eps_norm)  # (..., 1)

    # trace와 cosθ 계산: cosθ = (trace(R)-1)/2
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = th.clamp(cos_theta, -1.0, 1.0)

    # sinθ = norm_r/2  (참고: r_vec = 2 sinθ · axis)
    sin_theta = norm_r / 2

    # theta를 atan2로 계산: atan2(sinθ, cosθ)
    # cos_theta의 shape (..., )이므로 unsqueeze하여 (..., 1)로 맞춥니다.
    theta = th.atan2(sin_theta, cos_theta.unsqueeze(-1))  # (..., 1)

    # 보통은 factor = theta/(norm_r) 로 계산해야 하나, norm_r가 0에 가까우면 문제가 됩니다.
    # 그래서 norm_r가 충분히 크면 regular한 값을, 그렇지 않으면 Taylor 전개에 따른 근사값으로 smooth하게 보간합니다.
    eps = 1e-12 # 1e-6  # 임계값
    # regular branch: 분모에 eps를 더해 안전하게 계산 (norm_r가 충분히 크면 eps의 영향은 미미함)
    factor_reg = theta / (norm_r + eps)
    # 작은 각에 대한 Taylor 근사: theta/(2 sinθ) ~ 0.5 + theta^2/12 + theta^4/120  (θ->0)
    factor_taylor = 0.5 + theta**2 / 12 + theta**4 / 120

    # 보간 weight: norm_r가 eps 이상이면 1, eps 미만이면 norm_r/eps로 0~1 사이의 값을 취함.
    weight = (norm_r / eps).clamp(max=1.0)
    # smooth하게 보간
    factor = weight * factor_reg + (1 - weight) * factor_taylor

    # 최종 axis-angle: r = factor * (R - R^T)^\vee
    angle_axis = factor * r_vec
    return angle_axis

def rotaa2rotmat(rotaa):
    '''
    :param rotaa: (*size, 3)
    :return rotmat: (*size, 3, 3)
    '''
    assert rotaa.shape[-1] == 3
    rotmat = axis_angle_to_rotation_matrix(rotaa)
    return rotmat

def rotmat2rotaa(rotmat):
    '''
    :param rotmat: (*size, 3, 3)
    :return rotaa (*size, 3)
    '''
    assert rotmat.shape[-1] == 3 and rotmat.shape[-2] == 3
    rotaa = rotation_matrix_to_axis_angle(rotmat)
    return rotaa

def rot6d2rotmat(rot6d):
    '''
    :param rot6d: (*size, 6)
    :return rotmat: (*size, 3, 3)
    '''
    assert rot6d.shape[-1] == 6
    size = rot6d.shape[:-1]
    x_raw = rot6d[..., 0:3]
    y_raw = rot6d[..., 3:6]
    x = F.normalize(x_raw, dim=-1)
    z = th.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = th.cross(z, x, dim=-1)
    rotmat = th.stack((x,y,z), dim=-2)
    return rotmat

def rotmat2rot6d(rotmat):
    '''
    :param rotmat: (*size, 3, 3)
    :return rot6d: (*size, 6)
    '''
    assert rotmat.shape[-1] == 3 and rotmat.shape[-2] == 3
    size = rotmat.shape[:-2]
    rot6d = rotmat[..., :2, :].reshape(*size, 6)
    return rot6d

def rotaa2rot6d(rotaa):
    '''
    :param rotaa: (*size, 3)
    :return rot6d: (*size, 6)
    '''
    assert rotaa.shape[-1] == 3
    size = rotaa.shape[:-1]
    rotmat = axis_angle_to_rotation_matrix(rotaa)
    rot6d = rotmat[..., :2, :].reshape(*size, 6)
    return rot6d

def rot6d2rotaa(rot6d):
    '''
    :param rot6d: (*size, 6)
    :return rotaa: (*size, 3)
    '''
    assert rot6d.shape[-1] == 6
    size = rot6d.shape[:-1]
    x_raw = rot6d[...,0:3]
    y_raw = rot6d[...,3:6]
    x = F.normalize(x_raw, dim=-1)
    z = th.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = th.cross(z, x, dim=-1)
    rotmat = th.stack((x,y,z), dim=-2)
    rotaa = rotation_matrix_to_axis_angle(rotmat)
    return rotaa

def compute_se3(rotmat, transl3d):
    '''
    :param rotmat: (*size, 3, 3)
    :param transl3d: (*size, 3)
    :return se3: (*size, 4, 4)
    '''
    assert rotmat.shape[-1] == 3 and rotmat.shape[-2] == 3
    assert transl3d.shape[-1] == 3
    assert rotmat.shape[:-2] == transl3d.shape[:-1]
    size = rotmat.shape[:-2]
    transl3d = transl3d.view(*size, 3, 1)
    upper_part = th.cat((rotmat, transl3d), dim=-1) # (*size, 3, 4)
    last_row = th.tensor([0, 0, 0, 1], dtype=rotmat.dtype, device=rotmat.device)
    last_row = last_row.view(1, 1, 4).expand(*size, 1, 4)  # (*size, 1, 4)
    se3 = th.cat((upper_part, last_row), dim=-2)  # (*size, 4, 4)
    return se3

def decompose_se3(se3):
    '''
    :param se3: (*size, 4, 4)
    :return rotmat: (*size, 3, 3)
    :return transl3d: (*size, 3)
    '''
    assert se3.shape[-1] == 4 and se3.shape[-2] == 4, "Input must be a valid SE(3) matrix with shape (*size, 4, 4)"
    
    # Extract rotation matrix: (*size, 3, 3)
    rotmat = se3[..., :3, :3]
    
    # Extract translation vector: (*size, 3)
    transl3d = se3[..., :3, 3]
    
    return rotmat, transl3d

def inverse_transfrom_sampling(cdf: th.Tensor, continuous_random_variables: th.Tensor, uniform_samples: th.Tensor) -> th.Tensor:
    assert len(continuous_random_variables.shape) == 1
    assert cdf.shape[-1] == len(continuous_random_variables)
    assert 0 <= th.min(uniform_samples).item() and th.max(uniform_samples).item() <= 1
    assert uniform_samples.shape[:-1] == cdf.shape[:-1]
    
    indices = th.searchsorted(cdf, uniform_samples, right=False)
    indices = th.clamp(indices - 1, min=0, max=len(continuous_random_variables) - 2) # RIP
    cdf0 = th.gather(cdf, dim=-1, index=indices)
    cdf1 = th.gather(cdf, dim=-1, index=indices + 1)
    crv0 = continuous_random_variables[indices]
    crv1 = continuous_random_variables[indices + 1]

    t = (uniform_samples - cdf0) / ((cdf1 - cdf0) + (cdf1 == cdf0).float()) # 0 case handling
    
    return crv0 + t * (crv1 - crv0)

class IsotrophicGaussianDistribution:
    def __init__(self, device: str, dtype=th.float32, linspace_points=100000, epsilon_std=0.5):
        """
        Args:
            linspace_points (int, optional): approximation accuracy to compute cdf. Defaults to 100000.
            epsilon_std (int, optional): epsilon value for standard isotrophic Gaussian distribution. Defaults to 0.5.
        """
        self.device = device
        self.dtype = dtype
        self.epsilon_std = epsilon_std
        
        omega_start = 0 + 1e-8
        omega_stop = th.pi
        self.omega_values = th.linspace(omega_start, omega_stop, linspace_points + 1, dtype=dtype, device=self.device)[:-1]  # Rotation angles in (-π, π]

    def pdf_omega_approx(self, epsilon: th.Tensor, omega: th.Tensor) -> th.Tensor:
        '''
        Computes the heat kernel on SO(3) using the closed-form approximation for small epsilon with norm factor.

        :param epsilon: parameter corresponding to the standard deviation of the Gaussian distribution (*size_epsilon)
        :param omega: Rotation angle in radians (*size_omega)
        :return: probability density (*size_epsilon, *size_omega)
        '''

        assert epsilon.min().item() > 0, "epsilon must be positive"
        assert epsilon.max().item() <= 1 / self.epsilon_std, f"epsilon must be less than {1 / self.epsilon_std} due to approximation error"
        assert omega.min().item() > 0, "omega must be positive"
        assert omega.max().item() <= th.pi, "omega must be less than pi"

        size_omega = omega.shape
        size_epsilon = epsilon.shape
        epsilon = epsilon.reshape(*size_epsilon, *((1,)*len(size_omega)))
        omega = omega.reshape(*((1,)*len(size_epsilon)), *size_omega)

        epsilon = epsilon * self.epsilon_std # GU
        
        norm_factor = (1 - th.cos(omega)) / th.pi
        sqrt_term = th.sqrt(th.tensor(th.pi)) * epsilon**-3 * th.exp(epsilon**2 / 4 - (omega / 2)**2 / epsilon**2)
        sin_term = 2 * th.sin(omega / 2)
        exponentials = ((omega - 2 * th.pi) * th.exp(th.pi * (omega - th.pi) / epsilon**2) +
                        (omega + 2 * th.pi) * th.exp(-th.pi * (omega + th.pi) / epsilon**2))
        correction_term = omega - exponentials

        return norm_factor * sqrt_term * correction_term / sin_term
    
    def sample(self, size: tuple, epsilon: th.Tensor) -> th.Tensor:
        '''
        :param size: size of the output tensor
        :param epsilon: parameter corresponding to the standard deviation of the Gaussian distribution (*size_epsilon)
        :return: axis-angle samples (*size_epsilon, *size, 3)        
        '''
        
        assert epsilon.min().item() >= 0, "epsilon must be positive"
        assert epsilon.max().item() <= 1 / self.epsilon_std, f"epsilon must be less than {1 / self.epsilon_std} due to approximation error"

        epsilon = epsilon.clone()
        size_epsilon = epsilon.shape
        
        # 0 case handling
        mask = epsilon != 0
        epsilon[~mask] = 0.01
        
        pdf_omega = self.pdf_omega_approx(epsilon, self.omega_values) # (*size_epsilon, linspace_points)
        pdf_omega_normalized = pdf_omega / th.trapz(pdf_omega, self.omega_values).reshape(*size_epsilon, 1) # (*size_epsilon, linspace_points)
        cdf_omega = th.cumsum(pdf_omega_normalized, dim=-1) * (self.omega_values[1] - self.omega_values[0]) # (*size_epsilon, linspace_points)
        cdf_omega[..., -1] = 1.0
        
        uniform_samples = th.rand(size_epsilon + size, dtype=self.dtype, device=self.device).reshape(*size_epsilon, -1) # (*size_epsilon, num_samples)
        omega_samples = inverse_transfrom_sampling(cdf_omega, self.omega_values, uniform_samples).reshape(*size_epsilon, *size, 1)
        
        uniform_samples = th.randn(size_epsilon + size + (3,), device=self.device) # RIP
        axis_samples = uniform_samples / (th.linalg.norm(uniform_samples, dim=-1).reshape(*size_epsilon, *size, 1) + 1e-8)
        
        aa_samples = axis_samples * omega_samples # (*size_epsilon, *size, 3)
        
        # 0 case handling
        aa_samples = aa_samples * mask.reshape(*size_epsilon, *((1,)*len(size)), 1)
        
        return aa_samples
    
    def sample_from_approx_std(self, size: tuple, epsilon: th.Tensor) -> th.Tensor:
        '''
        :param size: size of the output tensor
        :param epsilon: parameter corresponding to the standard deviation of the Gaussian distribution (*size_epsilon)
        :return: axis-angle samples (*size_epsilon, *size, 3)        
        '''
        
        assert epsilon.min().item() >= 0, "epsilon must be positive"
        assert epsilon.max().item() <= 1 / self.epsilon_std, f"epsilon must be less than {1 / self.epsilon_std} due to approximation error"

        epsilon = epsilon.clone()
        size_epsilon = epsilon.shape
        
        # 0 case handling
        mask = epsilon != 0
        epsilon[~mask] = 0.01

        pdf_omega = self.pdf_omega_approx(epsilon, self.omega_values) # (*size_epsilon, linspace_points)
        pdf_omega_normalized = pdf_omega / th.trapz(pdf_omega, self.omega_values).reshape(*size_epsilon, 1) # (*size_epsilon, linspace_points)
        cdf_omega = th.cumsum(pdf_omega_normalized, dim=-1) * (self.omega_values[1] - self.omega_values[0]) # (*size_epsilon, linspace_points)
        cdf_omega[..., -1] = 1.0
        
        uniform_samples = th.rand(size_epsilon + size, dtype=self.dtype, device=self.device).reshape(*size_epsilon, -1) # (*size_epsilon, num_samples)
        omega_samples = inverse_transfrom_sampling(cdf_omega, self.omega_values, uniform_samples).reshape(*size_epsilon, *size, 1)
        
        uniform_samples = th.randn(size_epsilon + size + (3,), device=self.device) # RIP
        axis_samples = uniform_samples / (th.linalg.norm(uniform_samples, dim=-1).reshape(*size_epsilon, *size, 1) + 1e-8)

        aa_samples = axis_samples * omega_samples # (*size_epsilon, *size, 3)
        
        # 0 case handling
        aa_samples = aa_samples * mask.reshape(*size_epsilon, *((1,)*len(size)), 1)
        
        # return aa_samples / (epsilon.reshape(*size_epsilon, *([1]*len(size)), 1) + 1e-8) # GU
        
        aa_samples = aa_samples / (epsilon.reshape(*size_epsilon, *([1]*len(size)), 1))
            
        return aa_samples