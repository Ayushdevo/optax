:github_url: https://github.com/google-deepmind/optax/tree/main/docs

Building a custom gradient transformation
=========================================

Optax optimizers are built from small, composable gradient transformations.
A transformation has two functions:

* `init(params)` creates its state from a parameter pytree.
* `update(updates, state, params=None)` returns a pair
  `(transformed_updates, new_state)`.

The `updates` argument is usually a gradient pytree. Its leaves have the
same structure as the parameters. Transformations can be combined using
:func:`optax.chain`.

A stateless transformation
--------------------------

This example clips **each individual gradient element** into a fixed range.
It is intentionally distinct from :func:`optax.clip_by_global_norm`, which
scales the entire pytree based on its overall norm.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import optax

   def clip_elements(max_abs_value: float) -> optax.GradientTransformation:
       if max_abs_value <= 0:
           raise ValueError("max_abs_value must be positive")

       def init_fn(params):
           del params  # This transformation does not maintain any state.
           return optax.EmptyState()

       def update_fn(updates, state, params=None):
           del params  # No parameter values are needed.
           clipped = jax.tree.map(
               lambda gradient: jnp.clip(
                   gradient, -max_abs_value, max_abs_value
               ),
               updates,
           )
           return clipped, state

       return optax.GradientTransformation(init_fn, update_fn)

   params = {"weight": jnp.array([1.0, -1.0])}
   gradients = {"weight": jnp.array([3.0, -0.25])}

   transformation = clip_elements(0.5)
   state = transformation.init(params)
   clipped, state = transformation.update(gradients, state, params)

   assert jnp.allclose(clipped["weight"], jnp.array([0.5, -0.25]))

A transformation's output must preserve the expected pytree structure. Even
when no state is needed, `init` should return a state object and `update`
must return it alongside the transformed updates.

Using the transformation in an optimizer
----------------------------------------

Apply the clip before the learning-rate scale, then update the parameters.
The order of transformations in `chain` matters: the output of one is the
input of the next.

.. code-block:: python

   optimizer = optax.chain(clip_elements(0.5), optax.scale(-0.1))
   state = optimizer.init(params)
   updates, state = optimizer.update(gradients, state, params)
   new_params = optax.apply_updates(params, updates)

   assert jnp.allclose(
       new_params["weight"], jnp.array([0.95, -0.975])
   )

For transformations that accumulate statistics (such as momentum), replace
`optax.EmptyState` with a state containing those statistics and update it
on each call. `params` is optional because some transformations need only
the gradients, while others also depend on current parameter values.

See also :doc:`api/transformations` for the built-in building blocks and
:doc:`api/combining_optimizers` for composition utilities.
