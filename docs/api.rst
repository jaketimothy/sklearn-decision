API reference
=============

.. currentmodule:: sklearn_decision

Estimators
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   QuestionFeaturizer
   ChoiceEncoder
   NoulClassifier
   ChoiceClassifier
   ScoreRegressor

Calibration
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   calibrate_zero_shot

Questions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   noul
   choice
   score
   load_bank
   save_bank

Choice codebooks
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   choice_bank
   exemplar_options

Decision models
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DecisionModel
   Capabilities
   JevModel
   TransformersModel
   FakeModel
   register_model
   resolve_model

Answers and errors
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NoulAnswer
   DistAnswer
   Response
   DecisionModelError
   JevAPIError

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   grouped_permutation_importance
   clear_memory_cache
   clear_secret_cache
