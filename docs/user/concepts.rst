
============
Concepts
============

The Pipeline or Algorithm Tree
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the hearth of the cuvis.ai framework is the Pipeline class.
This class allows the easy definition and execution of machine learning pipelines,
using both shallow learning and deep learning modules.
In its simple form the Pipeline is a list of modules that will process the data one after another.
This modules can be of different types.
For the pipeline this are Preprocessor nodes, unsupervised nodes and supervised nodes.

Preprocessor Nodes
^^^^^^^^^^^^^^^^^^

This type of modules represent all Modules that do not need labeled data to be trained,
but rather are trained on the input data themselves.
Those modules are often used for tasks like dimensionality reduction (e.g. PCA) or similar things.

Supervised Nodes
~~~~~~~~~~~~~~~~

Unsupervised Nodes
~~~~~~~~~~~~~~~~~~

Transformation Nodes
~~~~~~~~~~~~~~~~~~~~

Distance Nodes
~~~~~~~~~~~~~~

Decider Nodes
~~~~~~~~~~~~~