import numpy as np
from text_diffusion.noise_schedule import LinearNoiseSchedule


def test_linear_noise_schedule():
    # GIVEN
    schedule = LinearNoiseSchedule()
    batch_size, seq_len, embed_dim = 2, 5, 3
    x = np.random.randn(batch_size, seq_len, embed_dim)
    noise_0 = np.zeros((batch_size, seq_len, embed_dim))
    noise_1 = np.ones((batch_size, seq_len, embed_dim))
    # WHEN
    x0_noisy = schedule.add_noise(x, noise_0, 0)
    x1_noisy = schedule.add_noise(x, noise_1, 9_999)
    # THEN
    np.testing.assert_allclose(x0_noisy, 0.9999 * x, rtol=1e-4)
    np.testing.assert_allclose(x1_noisy, noise_1, rtol=1e-10)
