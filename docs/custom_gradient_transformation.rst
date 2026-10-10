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

A stateful transformation
-------------------------

A stateful transformation can retain information from earlier steps. The
example below maintains an exponential moving average of gradients and returns
that running average as the update. It demonstrates how the state is carried
between calls; for production momentum, prefer :func:`optax.trace`.

.. code-block:: python

   from typing import NamedTuple

   class MovingAverageState(NamedTuple):
       average: dict

   def running_average(decay: float) -> optax.GradientTransformation:
       if not 0 <= decay < 1:
           raise ValueError("decay must be in [0, 1)")

       def init_fn(params):
           return MovingAverageState(
               average=jax.tree.map(jnp.zeros_like, params)
           )

       def update_fn(updates, state, params=None):
           del params
           average = jax.tree.map(
               lambda previous, gradient: (
                   decay * previous + (1 - decay) * gradient
               ),
               state.average,
               updates,
           )
           return average, MovingAverageState(average=average)

       return optax.GradientTransformation(init_fn, update_fn)

   tx = running_average(0.5)
   state = tx.init({"weight": jnp.array([0.0])})
   first, state = tx.update(
       {"weight": jnp.array([2.0])}, state
   )
   second, state = tx.update(
       {"weight": jnp.array([4.0])}, state
   )

   assert jnp.allclose(first["weight"], jnp.array([1.0]))
   assert jnp.allclose(second["weight"], jnp.array([2.5]))

Initializing state with `jax.tree.map` lets the example work with nested
parameter pytrees. An update step should return a new state instead of
mutating the old one, which also makes the transformation suitable for
functional JAX training loops.

Nested pytree support
---------------------

A transformation should preserve every leaf, including nested dictionaries.

.. code-block:: python

   nested = {
       "encoder": {"weight": jnp.array([2.0, -2.0])},
       "bias": jnp.array([0.2]),
   }
   tx = clip_elements(0.5)
   clipped, _ = tx.update(nested, tx.init(nested))
   assert jnp.allclose(clipped["encoder"]["weight"], [0.5, -0.5])
   assert jnp.allclose(clipped["bias"], [0.2])

Empty pytrees
-------------

For a stateless, elementwise transformation, an empty pytree can be processed
without allocating any state or changing the tree structure.

.. code-block:: python

   empty = {}
   tx = clip_elements(0.5)
   result, state = tx.update(empty, tx.init(empty))
   assert result == {}
   assert isinstance(state, optax.EmptyState)

Zero gradients
--------------

Clipping should preserve zero entries exactly, including leaves that represent
currently frozen parameters.

.. code-block:: python

   tx = clip_elements(0.5)
   zeros = {"frozen": jnp.zeros((2, 3))}
   result, _ = tx.update(zeros, tx.init(zeros))
   assert jnp.array_equal(result["frozen"], zeros["frozen"])

JIT compilation
---------------

The transformation update can be JIT-compiled as long as its arguments have
compatible pytree structures and array shapes.

.. code-block:: python

   tx = clip_elements(0.5)
   params = {"weight": jnp.array([1.0, -1.0])}
   state = tx.init(params)
   compiled_update = jax.jit(tx.update)
   clipped, state = compiled_update(
       {"weight": jnp.array([2.0, -2.0])}, state, params
   )
   assert jnp.allclose(clipped["weight"], [0.5, -0.5])

Transformation order
--------------------

Changing the order in a chain can change the result. Clipping before scaling
is not generally equivalent to clipping afterward.

.. code-block:: python

   params = {"weight": jnp.array([0.0])}
   gradients = {"weight": jnp.array([4.0])}

   clip_then_scale = optax.chain(clip_elements(0.5), optax.scale(2))
   scale_then_clip = optax.chain(optax.scale(2), clip_elements(0.5))

   first, _ = clip_then_scale.update(
       gradients, clip_then_scale.init(params)
   )
   second, _ = scale_then_clip.update(
       gradients, scale_then_clip.init(params)
   )
   assert jnp.allclose(first["weight"], [1.0])
   assert jnp.allclose(second["weight"], [0.5])

Resetting state
---------------

Calling `init` again starts a fresh transformation history. This matters when
restarting training or intentionally discarding accumulated statistics.

.. code-block:: python

   tx = running_average(0.5)
   params = {"weight": jnp.array([0.0])}
   gradient = {"weight": jnp.array([2.0])}
   state = tx.init(params)
   _, state = tx.update(gradient, state)
   next_value, _ = tx.update(gradient, state)
   reset_value, _ = tx.update(gradient, tx.init(params))
   assert jnp.allclose(next_value["weight"], [1.5])
   assert jnp.allclose(reset_value["weight"], [1.0])

Decay boundary
--------------

With zero decay, the running-average example returns the latest gradient.
This is also useful as a control case for checking the stateful implementation.

.. code-block:: python

   tx = running_average(0.0)
   params = {"weight": jnp.array([0.0])}
   state = tx.init(params)
   _, state = tx.update({"weight": jnp.array([2.0])}, state)
   current, state = tx.update(
       {"weight": jnp.array([-3.0])}, state
   )
   assert jnp.allclose(current["weight"], [-3.0])

Parameter structure
-------------------

Gradient transformations normally receive pytrees with the same structure as
the initialized parameters. Check that structure before applying updates
when writing custom loops that may dynamically freeze or replace parameters.

.. code-block:: python

   params = {"dense": {"weight": jnp.zeros((2, 3))}}
   gradients = {"dense": {"weight": jnp.ones((2, 3))}}
   assert jax.tree.structure(params) == jax.tree.structure(gradients)

   tx = clip_elements(0.5)
   clipped, _ = tx.update(gradients, tx.init(params))
   assert clipped["dense"]["weight"].shape == (2, 3)

Threading state through a training loop
---------------------------------------

Always use the new state from `update`. Reusing the initial state on every
step turns a stateful transformation into an unintended one-step operation.

.. code-block:: python

   tx = running_average(0.5)
   state = tx.init({"weight": jnp.array([0.0])})
   history = []
   for value in (2.0, 4.0, 8.0):
       updates = {"weight": jnp.array([value])}
       transformed, state = tx.update(updates, state)
       history.append(float(transformed["weight"][0]))

   assert jnp.allclose(jnp.array(history), [1.0, 2.5, 5.25])

Full optimizer step
-------------------

In a training step, compute gradients, transform them, and use
:func:`optax.apply_updates` to produce new parameters.

.. code-block:: python

   def squared_error(params):
       return jnp.sum((params["weight"] - 3.0) ** 2)

   tx = optax.chain(clip_elements(0.5), optax.scale(-0.1))
   params = {"weight": jnp.array([0.0])}
   state = tx.init(params)
   gradients = jax.grad(squared_error)(params)
   updates, state = tx.update(gradients, state, params)
   new_params = optax.apply_updates(params, updates)
   assert jnp.allclose(new_params["weight"], [0.05])

See also :doc:`api/transformations` for the built-in building blocks and
:doc:`api/combining_optimizers` for composition utilities.
