# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Finite-difference contracts for differentiating through optimizer rollouts.

Convergence alone does not detect an accidentally detached optimizer state or
learning rate. Compare both AD modes with a numerical Jacobian of the final
parameters after eight training steps, including every initial parameter and
the learning rate. Piecewise optimizers are tested away from branch boundaries;
this does not assert differentiability at sign, maximum or threshold changes.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import alias
from optax._src import update
from optax._src import transform


class OptimizerHypergradientTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Restore the caller's dtype setting even when an assertion fails.
    previous_x64 = jax.config.x64_enabled
    self.addCleanup(jax.config.update, 'jax_enable_x64', previous_x64)
    jax.config.update('jax_enable_x64', True)

  def _check_rollout(self, factory, kwargs, jit):
    # Dense features and a quartic penalty give cross-coordinate curvature.
    features = jnp.array([[1.0, 0.3], [-0.2, 1.4], [0.7, -0.8]])
    targets = jnp.array([0.2, -0.5, 0.8])

    def loss(params):
      residual = features @ params['weights'] + params['bias'] - targets
      return jnp.mean(residual**2) + 0.07 * jnp.sum(params['weights'] ** 4)

    def rollout(inputs):
      params = {'weights': inputs[:2], 'bias': inputs[2]}
      optimizer = factory(learning_rate=inputs[3], **kwargs)

      def step(carry, _):
        params, state = carry
        updates, state = optimizer.update(jax.grad(loss)(params), state, params)
        return (update.apply_updates(params, updates), state), None

      (params, _), _ = jax.lax.scan(
          step, (params, optimizer.init(params)), None, length=8
      )
      return jnp.concatenate((params['weights'], params['bias'][None]))

    inputs = jnp.array([1.4, -0.8, 0.3, 0.06])
    # Numerical differentiation uses only primal values, independently of AD.
    primal = jax.jit(rollout)
    columns = []
    for direction in jnp.eye(inputs.size):
      # Two radii guard against crossing a piecewise branch or cancellation.
      estimates = []
      for h in (1e-4, 5e-5):
        estimates.append(
            (primal(inputs + h * direction) - primal(inputs - h * direction))
            / (2 * h)
        )
      np.testing.assert_allclose(
          estimates[0], estimates[1], rtol=2e-5, atol=2e-7
      )
      columns.append(estimates[1])
    expected = jnp.stack(columns, axis=1)
    self.assertGreater(float(jnp.linalg.norm(expected[:, 3])), 1e-4)
    for differentiate in (jax.jacfwd, jax.jacrev):
      derivative = differentiate(rollout)
      if jit:
        derivative = jax.jit(derivative)
      actual = derivative(inputs)
      self.assertTrue(bool(jnp.all(jnp.isfinite(actual))))
      np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)

  @parameterized.parameters(False, True)
  def test_radam(self, jit):
    # Eight steps exercise both sides of the rectification threshold.
    self._check_rollout(
        alias.radam, {'b1': 0.7, 'b2': 0.8, 'eps_root': 0.0001}, jit
    )

  @parameterized.product(
      dtype=(jnp.float32, jnp.float64), nesterov=(False, True)
  )
  def test_radam_batched_reverse_derivatives(self, dtype, nesterov):
    # vmap batches both unrectified and rectified counts in one update.
    def rescale(gradients, count):
      optimizer = transform.scale_by_radam(
          b1=0.7, b2=0.8, eps_root=1e-4, nesterov=nesterov
      )
      state = optimizer.init(gradients)._replace(count=count)
      return optimizer.update(gradients, state)[0].sum()

    gradients = jnp.array([0.3, 0.5], dtype=dtype)
    counts = jnp.arange(8, dtype=jnp.int32)
    reverse = jax.jit(jax.vmap(jax.grad(rescale), in_axes=(None, 0)))
    actual = reverse(gradients, counts)
    self.assertTrue(bool(jnp.all(jnp.isfinite(actual))))
    # Each output coordinate depends only on its corresponding input gradient.
    h = 1e-3 if dtype == jnp.float32 else 1e-5
    primal = jax.jit(jax.vmap(rescale, in_axes=(None, 0)))
    expected = jnp.stack(
        [
            (
                primal(gradients + h * d, counts)
                - primal(gradients - h * d, counts)
            )
            / (2 * h)
            for d in jnp.eye(2, dtype=dtype)
        ],
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=1e-4)

  @parameterized.parameters(False, True)
  def test_sgd(self, jit):
    # Nesterov momentum must carry sensitivity between steps.
    self._check_rollout(alias.sgd, {'momentum': 0.7, 'nesterov': True}, jit)

  @parameterized.parameters(False, True)
  def test_adam(self, jit):
    # Both debiased moments participate in the training derivative.
    self._check_rollout(
        alias.adam, {'b1': 0.7, 'b2': 0.8, 'eps_root': 0.0001}, jit
    )

  @parameterized.parameters(False, True)
  def test_adamw(self, jit):
    # Decoupled decay also depends on the current parameters.
    self._check_rollout(
        alias.adamw,
        {'b1': 0.7, 'b2': 0.8, 'eps_root': 0.0001, 'weight_decay': 0.2},
        jit,
    )

  @parameterized.parameters(False, True)
  def test_adabelief(self, jit):
    # The residual-based variance must remain differentiable.
    self._check_rollout(
        alias.adabelief, {'b1': 0.7, 'b2': 0.8, 'eps_root': 0.0001}, jit
    )


if __name__ == '__main__':
  absltest.main()
