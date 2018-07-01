# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# **************************************** ContrastiveHebbianMechanism *************************************************

"""
.. _ContrastiveHebbian_Overview:

Overview
--------

A ContrastiveHebbianMechanism is a subclass of `RecurrentTransferMechanism` that is customized for use with the
Contrastive Hebbian learning rule.  See the following references for a description of the learning rule,
its relationship to the backpropagation learning rule, and its use in connectionist networks:

  `Movellan, J. R. (1991). Contrastive Hebbian learning in the continuous Hopfield model. In Connectionist Models
  (pp. 10-17) <https://www.sciencedirect.com/science/article/pii/B978148321448150007X>`_

  `Xie, X., & Seung, H. S. (2003). Equivalence of backpropagation and contrastive Hebbian learning in a layered network.
  Neural computation, 15(2), 441-454 <https://www.mitpressjournals.org/doi/abs/10.1162/089976603762552988>`_

  `O'reilly, R. C. (2001). Generalization in interactive networks: The benefits of inhibitory competition and Hebbian
  learning. Neural computation, 13(6), 1199-1241 <https://www.mitpressjournals.org/doi/abs/10.1162/08997660152002834>`_

  `Verguts, T., & Notebaert, W. (2008). Hebbian learning of cognitive control: dealing with specific and nonspecific
  adaptation. Psychological review, 115(2), 518 <http://psycnet.apa.org/record/2008-04236-010>`_

The features and operation of a ContrastiveHebbianMechanism that differ from thoe of a RecurrentTransferMechanism are
described below.

.. _ContrastiveHebbian_Creation:

Creation
--------

When a ContrastiveHebbianMechanism is created, its `has_recurrent_input_state
<RecurrentTransferMechanism.has_recurrent_input_state>` attribute is automatically assigned as `True`, and is
automatically assigned two of its four `Standard OutputStates <ContrastiveHebbianMechanism_Standard_OutputStates>`:
*CURRENT_ACTIVITY_OUTPUT*, *ACTIVITY_DIFFERENT_OUTPUT* (see `below <ContrastiveHebbian_Structure>`). Additional
OutputStates can be specified in the **additional_output_states** argument of its constructor.  It uses the same
default `function <ContrastiveHebbianMechanism.function>` as a `RecurrentTransferMechanism`, but its `integrator_mode
<ConstrativeHebbianMechanism.integrator_mode>` is automatically set to `True`.  If a ContrastiveHebbianMechanism is
`configured for learning <Recurrent_Transfer_Learning>`, it is assigned `ContrastiveHebbian` as its `learning_function
<ContrastiveHebbian.learning_function>`.  The **convergence_function** and **convergence** criterion** arguments of
its constructor specify, respectively, the function and criterion used for determining when its `execution
<ContrastiveHebbian_Execution>` is complete.

.. _ContrastiveHebbian_Structure:

Structure
---------

.. _ContrastiveHebbian_Input:

Input
~~~~~

A ContrastiveHebbianMechanism is automatically assigned two `InputStates <InputState>` on creation: *RECURRENT* and
*EXTERNAL* (that is, its `has_recurrent_input_state <RecurrentTransferMechanism.has_recurrent_input_state>` attribute
is automatically assigned as `True`),  This is so that the input from its `recurrent_projection
<RecurrentTransferMechanism.recurrent_projection>` can be kept separate from its external input during the
`plus phase <ContrastiveHebbianMechanism_Inputs>` and `minus phase <ContrastiveHebbianMechanism_Inputs>` of its
operation (see `ContrastiveHebbian_Execution`).

.. _ContrastiveHebbian_Functions:

Functions
~~~~~~~~~

* `function <ContrastiveHebbianMechanism.function>` -- the default is `Linear`, but it can be replaced with any
  function that satisfies the constraints of a `TransferMechanism's function <TransferMechanism_Function>`.
  This is used in conjunction with its `integrator_function <ContrastiveHebbianMechanism.integrator_function>` to
  update the activity of the ContrastiveHebbianMechanism.  Note that the `integrator_mode
  <ContrastiveHebbianMechanism.integrator_mode>` is always `True`.
..
* `convergence_function <ContrastiveHebbianMechanism.convergence_function>` -- coupled with the
  `convergence_criterion <ContrastiveHebbianMechanism.convergence_criterion>` attribute, this determines when the
  `execution <ContrastiveHebbian_Execution>` of the ContrastiveHebbianMechanism is complete.
..
* `learning_function <ContrastiveHebbianMechanism.learning_function>` -- the default is `ContrastiveHebbian`, but it
  can be replaced by any function that takes two 1d arrays ("activity states") and compares them to determine the
  `matrix <MappingProjection.matrix>` of its `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`.


.. _ContrastiveHebbian_Output:

Output
~~~~~~

ATTRIBUTES:
current_activity
plus_phase_activity
minus_phase_activity

LIST STANDARD OUTPUTSTATES
it is also automatically assigned
`CURRENT_ACTIVITY_OUTPUT`, `ACTIVITY_DIFFERENCE_OUTPUT` `PLUS_PHASE_OUTPUT` and `MINUS_PHASE_OUTPUT` `OutputStates
<OutputState>`.

A ConstrativeHebbianMechanism always use
has two `InputStates <InputState>` — `RECURRENT` and `EXTERNAL`; uses the `ContrastiveHebbian`Function as its
`function <ContrastiveHebbianMechanism.function>` by default; and has three `OutputStates <OutputState>` — 
`CURRENT_ACTIVITY_OUTPUT


.. _ContrastiveHebbian_Execution:

Execution
---------

COMMENT:
  - NOTE THAT IT IS ALWAYS RUN IN INTEGRATOR_MODE = TRUE
  - PLUS AND MINUS PHASES
  - CONVERGENCE
COMMENT

If it has been `configured for learning <Recurrent_Transfer_Learning>`
and is executed as part of a `System`, then its associated `LearningMechanism <AutoAssociativeLearningMechanism>` is
executed during the `execution phase <System_Execution>` of the System's execution.  Note that this is distinct from
the behavior of supervised learning algorithms (such as `Reinforcement` and `BackPropagation`), that are executed
during the `learning phase <System_Execution>` of a System's execution



.. _ContrastiveHebbian_Class_Reference:

Class Reference
---------------

"""

import numbers
from collections import Iterable

import numpy as np
import typecheck as tc
from enum import IntEnum

from psyneulink.components.functions.function import Function, Linear, is_function_type, ContrastiveHebbian, Distance
from psyneulink.components.states.outputstate import PRIMARY, StandardOutputStates
from psyneulink.globals.keywords import \
    CONTRASTIVE_HEBBIAN_MECHANISM, ENERGY, ENTROPY, FUNCTION, HOLLOW_MATRIX, \
    MAX_DIFF, MEAN, MEDIAN, NAME, RESULT, STANDARD_DEVIATION, VARIABLE, VARIANCE
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.utilities import is_numeric_or_none, parameter_spec
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism

__all__ = [
    'ConstrastiveHebbianError', 'ContrastiveHebbianMechanism', 'CONTRASTIVE_HEBBIAN_OUTPUT',
    'ACTIVITY_DIFFERENCE_OUTPUT', 'CURRENT_ACTIVITY_OUTPUT',
    'MINUS_PHASE_ACTIVITY', 'MINUS_PHASE_OUTPUT', 'PLUS_PHASE_ACTIVITY', 'PLUS_PHASE_OUTPUT'
]

CURRENT_ACTIVITY = 'current_activity'
PLUS_PHASE_ACTIVITY = 'plus_phase_activity'
MINUS_PHASE_ACTIVITY = 'minus_phase_activity'

CURRENT_ACTIVITY_OUTPUT = 'CURRENT_ACTIVITY_OUTPUT'
ACTIVITY_DIFFERENCE_OUTPUT = 'ACTIVITY_DIFFERENCE_OUTPUT'
PLUS_PHASE_OUTPUT = 'PLUS_PHASE_OUTPUT'
MINUS_PHASE_OUTPUT = 'MINUS_PHASE_OUTPUT'


class LearningPhase(IntEnum):
    MINUS = 1
    PLUS  = 0


# Used to index items of InputState.variable corresponding to recurrent and external inputs
RECURRENT = 0
EXTERNAL = -1


class ConstrastiveHebbianError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# This is a convenience class that provides list of standard_output_state names in IDE
class CONTRASTIVE_HEBBIAN_OUTPUT():

    """
        .. _ContrastiveHebbianMechanism_Standard_OutputStates:

        `Standard OutputStates <OutputState_Standard>` for `ContrastiveHebbianMechanism` (in addition to those
        for `RecurrentTransferMechanism` and `TransferMechanism`):

        .. _CURRENT_ACTIVITY_OUTPUT:

        *CURRENT_ACTIVITY_OUTPUT* : 1d np.array
            array of with current activity of the Mechanism.

        .. _ACTIVITY_DIFFERENCE_OUTPUT:

        *ACTIVITY_DIFFERENCE_OUTPUT* : 1d np.array
            array of element-wise differences in activity between the `plus and minus phases of execution
            <ContrastiveHebbianMechanism_Execution>`.

        .. _PLUS_PHASE_OUTPUT:

        *PLUS_PHASE_OUTPUT* : 1d np.array
            array of activity at the end of the `plus phase of execution <ContrastiveHebbianMechanism_Execution>`.

        .. _MINUS_PHASE_OUTPUT:

        *MINUS_PHASE_OUTPUT* : 1d np.array
            array of activity at the end of the `minus phase of execution <ContrastiveHebbianMechanism_Execution>`

        """
    CURRENT_ACTIVITY_OUTPUT=CURRENT_ACTIVITY_OUTPUT
    ACTIVITY_DIFFERENCE_OUTPUT=ACTIVITY_DIFFERENCE_OUTPUT
    PLUS_PHASE_OUTPUT=PLUS_PHASE_OUTPUT
    MINUS_PHASE_OUTPUT=MINUS_PHASE_OUTPUT


# IMPLEMENTATION NOTE:  IMPLEMENTS OFFSET PARAM BUT IT IS NOT CURRENTLY BEING USED
class ContrastiveHebbianMechanism(RecurrentTransferMechanism):
    """
    ContrastiveHebbianMechanism(                     \
    default_variable=None,                           \
    size=None,                                       \
    function=Linear,                                 \
    matrix=HOLLOW_MATRIX,                            \
    auto=None,                                       \
    hetero=None,                                     \
    initial_value=None,                              \
    noise=0.0,                                       \
    smoothing_factor=0.5,                            \
    clip=[float:min, float:max],                     \
    enable_learning=False,                           \
    learning_rate=None,                              \
    learning_function=ContrastiveHebbian,            \
    convergence_function=Distance(metric=MAX_DIFF),  \
    convergence_criterion=0.01,                      \
    additional_output_states=None,                   \
    params=None,                                     \
    name=None,                                       \
    prefs=None)

    Subclass of `RecurrentTransferMechanism` that implements a single-layer auto-recurrent network.

    COMMENT:
        Description
        -----------
            ContrastiveHebbianMechanism is a Subtype of RecurrentTransferMechanism customized to implement a
            the `ContrastiveHebbian` `LearningFunction`.
    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <ContrastiveHebbianMechanism.variable>` for
        `function <ContrastiveHebbianMechanism.function>`, and the `primary OutputState <OutputState_Primary>`
        of the Mechanism.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = ContrastiveHebbianMechanism(size = [3, 2])
            T2 = ContrastiveHebbian(default_variable = [[0, 0, 0], [0, 0]])

    function : TransferFunction : default Linear
        specifies the function used to transform the input;  can be `Linear`, `Logistic`, `Exponential`,
        or a custom function.

    matrix : list, np.ndarray, np.matrix, matrix keyword, or AutoAssociativeProjection : default HOLLOW_MATRIX
        specifies the matrix to use for creating a `recurrent AutoAssociativeProjection <ContrastiveHebbian_Structure>`,
        or an AutoAssociativeProjection to use.

        - If **auto** and **matrix** are both specified, the diagonal terms are determined by auto and the off-diagonal
          terms are determined by matrix.

        - If **hetero** and **matrix** are both specified, the diagonal terms are determined by matrix and the
          off-diagonal terms are determined by hetero.

        - If **auto**, **hetero**, and **matrix** are all specified, matrix is ignored in favor of auto and hetero.

    auto : number, 1D array, or None : default None
        specifies matrix as a diagonal matrix with diagonal entries equal to **auto**, if **auto** is not None;
        If **auto** and **hetero** are both specified, then matrix is the sum of the two matrices from **auto** and
        **hetero**.

        See **matrix** for details on how **auto** and **hetero** may overwrite matrix.

        Can be modified by control.

    hetero : number, 2D array, or None : default None
        specifies matrix as a hollow matrix with all non-diagonal entries equal to **hetero**, if **hetero** is not None;
        If **auto** and **hetero** are both specified, then matrix is the sum of the two matrices from **auto** and
        **hetero**.

        When diagonal entries of **hetero** are specified with non-zero values, these entries are set to zero before
        hetero is used to produce a matrix.

        See **hetero** (above) for details on how various **auto** and **hetero** specifications are summed to produce a
        matrix.

        See **matrix** (above) for details on how **auto** and **hetero** may overwrite matrix.

        Can be modified by control.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input.
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        a value added to the result of the `function <ContrastiveHebbianMechanism.function>`. See `noise
        <ContrastiveHebbianMechanism.noise>` for more details.

    smoothing_factor : float : default 0.5
        the smoothing factor for exponential time averaging of input::

         result = (smoothing_factor * variable) +
         (1-smoothing_factor * input to mechanism's function on the previous time step)

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <ContrastiveHebbianMechanism.function>` the item in
        index 0 specifies the minimum allowable value of the result, and the item in index 1 specifies the maximum
        allowable value; any element of the result that exceeds the specified minimum or maximum value is set to the
        value of `clip <ContrastiveHebbianMechanism.clip>` that it exceeds.


    enable_learning : boolean : default False
        specifies whether the Mechanism should be configured for learning;  if it is not (the default), then learning
        cannot be enabled until it is configured for learning by calling the Mechanism's `configure_learning
        <ContrastiveHebbianMechanism.configure_learning>` method.

    learning_rate : scalar, or list, 1d or 2d np.array, or np.matrix of numeric values: default False
        specifies the learning rate used by its `learning function <ContrastiveHebbianMechanism.learning_function>`.
        If it is `None`, the `default learning_rate for a LearningMechanism <LearningMechanism_Learning_Rate>` is
        used; if it is assigned a value, that is used as the learning_rate (see `learning_rate
        <ContrastiveHebbianMechanism.learning_rate>` for details).

    learning_function : function : default ContrastiveHebbian
        specifies the function for the LearningMechanism if `learning has been specified
        <ContrastiveHebbian_Learning>` for the ContrastiveHebbianMechanism.  It can be any function so long as it
        takes a list or 1d array of numeric values as its `variable <Function_Base.variable>` and returns a sqaure
        matrix of numeric values with the same dimensions as the length of the input.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <ContrastiveHebbianMechanism.name>`
        specifies the name of the ContrastiveHebbianMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the ContrastiveHebbianMechanism; see `prefs <ContrastiveHebbianMechanism.prefs>`
        for details.

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to Mechanism's `function <ContrastiveHebbianMechanism.variable>`.

    function : Function
        the Function used to transform the input.

    matrix : 2d np.array
        the `matrix <AutoAssociativeProjection.matrix>` parameter of the `recurrent_projection` for the Mechanism.

    recurrent_projection : AutoAssociativeProjection
        an `AutoAssociativeProjection` that projects from the Mechanism's `primary OutputState <OutputState_Primary>`
        back to its `primary inputState <Mechanism_InputStates>`.

    COMMENT:
       THE FOLLOWING IS THE CURRENT ASSIGNMENT
    COMMENT
    initial_value :  value, list or np.ndarray
        determines the starting value for time-averaged input (only relevant if `smoothing_factor
        <ContrastiveHebbianMechanism.smoothing_factor>` parameter is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    integrator_function:
        The `integrator_function <ContrastiveHebbianMechanism.integrator_function>` used by the Mechanism when it
        executes, which is an `AdaptiveIntegrator <AdaptiveIntegrator>`. Keep in mind that the `smoothing_factor
        <ContrastiveHebbianMechanism.smoothing_factor>` parameter of the `ContrastiveHebbianMechanism` corresponds to
        the `rate <ContrastiveHebbianMechanismIntegrator.rate>` of the `ContrastiveHebbianMechanismIntegrator`.

    COMMENT:
    ALWAYS TRUE;  MOVE THIS TO MODULE DOCSTRING
    integrator_mode:

        the variable of the mechanism is first passed into the following equation:

        .. math::
            value = previous\\_value(1-smoothing\\_factor) + variable \\cdot smoothing\\_factor + noise

        The result of the integrator function above is then passed into the `mechanism's function
        <ContrastiveHebbianMechanism.function>`. Note that on the first execution, *initial_value* sets previous_value.
    COMMENT

    noise : float or function
        value passed to the `integrator_function <ContrastiveHebbianMechanism.integrator_function>` that is added to
        the current input.

        If noise is a list or array, it must be the same length as `variable
        <ContrastiveHebbianMechanism.default_variable>`.

        If noise is specified as a single float or function, while `variable <ContrastiveHebbianMechanism.variable>`
        is a list or array, noise will be applied to each variable element. In the case of a noise function, this means
        that the function will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function
            (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value
            from its distribution on each execution. If noise is specified as a float or as a function with a fixed
            output, then the noise will simply be an offset that remains the same across all executions.

    smoothing_factor : float
        the smoothing factor for exponential time averaging of input when::

          result = (smoothing_factor * current input) + (1-smoothing_factor * result on previous time_step)

    clip : list [float, float]
        specifies the allowable range for the result of `function <ContrastiveHebbianMechanism.function>`

        the item in index 0 specifies the minimum allowable value of the result, and the item in index 1 specifies the
        maximum allowable value; any element of the result that exceeds the specified minimum or maximum value is set
        to the value of `clip <ContrastiveHebbianMechanism.clip>` that it exceeds.

    previous_input : 1d np.array of floats
        the value of the input on the previous execution, including the value of `recurrent_projection`.

    learning_enabled : bool
        indicates whether learning has been enabled for the ContrastiveHebbianMechanism.  It is set to `True` if
        `learning is specified <ContrastiveHebbian_Learning>` at the time of construction (i.e., if the
        **enable_learning** argument of the Mechanism's constructor is assigned `True`, or when it is configured for
        learning using the `configure_learning <ContrastiveHebbianMechanism.configure_learning>` method.  Once learning
        has been configured, `learning_enabled <RecurrentMechahinsm.learning_enabled>` can be toggled at any time to
        enable or disable learning; however, if the Mechanism has not been configured for learning, an attempt to
        set `learning_enabled <RecurrentMechahinsm.learning_enabled>` to `True` elicits a warning and is then
        ignored.

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values
        specifies the learning rate used by the `learning_function <ContrastiveHebbianMechanism.learning_function>`
        of the `learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>` (see `learning_rate
        <AutoAssociativeLearningMechanism.learning_rate>` for details concerning specification and default value
        assignement).

    learning_function : function
        the function used by the `learning_mechanism <ContrastiveHebbianMechanism.learning_mechanism>` to train the
        `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>` if `learning is specified
        <ContrastiveHebbian_Learning>`.

    learning_mechanism : LearningMechanism
        created automatically if `learning is specified <ContrastiveHebbian_Learning>`, and used to train the
        `recurrent_projection <ContrastiveHebbianMechanism.recurrent_projection>`.

    value : 2d np.array
        result of executing `function <ContrastiveHebbianMechanism.function>`; same value as first item of
        `output_values <ContrastiveHebbianMechanism.output_values>`.

    output_states : Dict[str: OutputState]
        an OrderedDict with the following `OutputStates <OutputState>` by default:

        * `CURRENT_ACTIVITY_OUTPUT`, the `value <OutputState.value>` of which is a 1d array with the current activity
          of the ContrastiveHebbianMechanism during execution.

        * `ACTIVITY_DIFFERENCE_OUTPUT`, the `value <OutputState.value>` of which is a 1d array with the element-wise
          differences in activity between the plus and minus phases of the last execution.

    output_values : List[1d np.array]
        a list with the following items by default:
        * **current_activity_output** at the end of an execution.
        * **activity_difference_output** at the end of an execution.

    name : str
        the name of the ContrastiveHebbianMechanism; if it is not specified in the **name** argument of the constructor,
        a default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ContrastiveHebbianMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    Returns
    -------
    instance of ContrastiveHebbianMechanism : ContrastiveHebbianMechanism

    """
    componentType = CONTRASTIVE_HEBBIAN_MECHANISM

    class ClassDefaults(RecurrentTransferMechanism.ClassDefaults):
        variable = np.array([[0]])

    paramClassDefaults = RecurrentTransferMechanism.paramClassDefaults.copy()

    standard_output_states = RecurrentTransferMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:CURRENT_ACTIVITY_OUTPUT,
                                    VARIABLE:CURRENT_ACTIVITY},
                                   {NAME:ACTIVITY_DIFFERENCE_OUTPUT,
                                    VARIABLE:[PLUS_PHASE_ACTIVITY, MINUS_PHASE_ACTIVITY],
                                    FUNCTION: lambda v: v[1] - v[0]},
                                   {NAME:PLUS_PHASE_OUTPUT,
                                    VARIABLE:PLUS_PHASE_ACTIVITY},
                                   {NAME:MINUS_PHASE_OUTPUT,
                                    VARIABLE:MINUS_PHASE_ACTIVITY},
                                   ])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(list, dict)) = None,
                 function=Linear,
                 matrix=HOLLOW_MATRIX,
                 auto=None,
                 hetero=None,
                 initial_value=None,
                 noise=0.0,
                 smoothing_factor: is_numeric_or_none=0.5,
                 clip=None,
                 enable_learning:bool=False,
                 learning_rate:tc.optional(tc.any(parameter_spec, bool))=None,
                 learning_function: tc.any(is_function_type) = ContrastiveHebbian,
                 convergence_function:tc.any(is_function_type)=Distance(metric=MAX_DIFF),
                 convergence_criterion:float=0.01,
                 additional_output_states:tc.optional(tc.any(str, Iterable))=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set=None):

        """Instantiate ContrastiveHebbianMechanism"""

        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        output_states = [CURRENT_ACTIVITY_OUTPUT, ACTIVITY_DIFFERENCE_OUTPUT]
        if additional_output_states:
            if isinstance(additional_output_states, list):
                output_states += additional_output_states
            else:
                output_states.append(additional_output_states)

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(convergence_function=convergence_function,
                                                  convergence_criterion=convergence_criterion,
                                                  output_states=output_states,
                                                  params=params)

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_states=input_states,
                         function=function,
                         matrix=matrix,
                         auto=auto,
                         hetero=hetero,
                         has_recurrent_input_state=True,
                         initial_value=initial_value,
                         noise=noise,
                         integrator_mode=True,
                         smoothing_factor=smoothing_factor,
                         clip=clip,
                         enable_learning=enable_learning,
                         learning_rate=learning_rate,
                         learning_function=learning_function,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs)

    def _instantiate_attributes_before_function(self, function=None, context=None):
        super()._instantiate_attributes_before_function(function=function, context=context)
        if isinstance(self.convergence_function, Function):
            self.convergence_function = self.convergence_function.function

    def _instantiate_attributes_after_function(self, context=None):

        # Assign these after instantiation of function, since they are initialized in _execute (see below)
        self.attributes_dict_entries.update({CURRENT_ACTIVITY:CURRENT_ACTIVITY,
                                             PLUS_PHASE_ACTIVITY:PLUS_PHASE_ACTIVITY,
                                             MINUS_PHASE_ACTIVITY:MINUS_PHASE_ACTIVITY})

        super()._instantiate_attributes_after_function(context=context)

    def _execute(self,
                 variable=None,
                 function_variable=None,
                 runtime_params=None,
                 context=None):

        if self.context.initialization_status == ContextFlags.INITIALIZING:
            # Set plus_phase and minus_phase activity vectors to zeros with size of an input projection
            self.current_activity = self.plus_phase_activity = self.minus_phase_activity = \
                self.input_state.socket_template
            self.learning_phase = None

        if self.learning_phase is None:
            self.learning_phase = LearningPhase.PLUS

        if self.learning_phase == LearningPhase.PLUS:
            self.finished = False
            self.plus_phase_activity = variable[EXTERNAL] + variable[RECURRENT]
            self.current_activity = self.plus_phase_activity
        else:
            self.minus_phase_activity = variable[RECURRENT]
            self.current_activity = self.minus_phase_activity

        value = super()._execute(variable=np.atleast_2d(self.current_activity),
                                 runtime_params=runtime_params,
                                 context=context)

        # Check for convergence
        previous_value = self.integrator_function.previous_value
        if (self.convergence_criterion is not None and
                abs(self.convergence_function([value, previous_value])) < self.convergence_criterion):

            # Terminate if this is the end of the minus phase
            if self.learning_phase == LearningPhase.MINUS:

                # ?? USE initial_value attribute below??
                self.is_finished = True
                # JDC: NOT SURE THIS IS THE CORRECT THING TO DO
                # self.input_states[RECURRENT].variable = self.output_states[PLUS_PHASE_OUTPUT].value
                self.input_states[RECURRENT].value = self.output_states[PLUS_PHASE_OUTPUT].value

            # JDC: NOT SURE THIS IS THE CORRECT THING TO DO;  MAYBE ONLY AT BEGINNING OF MINUS PHASE?
            # NOTE: "socket_template" is a convenience property = np.zeros(<InputState>.variable.shape[-1])
            # Initialize RECURRENT input to zero for next phase
            # self.input_state.variable[RECURRENT] = self.input_state.socket_template
            # self.input_states[RECURRENT].variable *= 0
            self.input_states[RECURRENT].value *= 0

            # Switch learning phase
            self.learning_phase = ~self.learning_phase

        return value

    @property
    def _learning_signal_source(self):
        '''Overrided default to use ACTIVITY_DIFFERENCE_OUTPUT as source of learning signal
        '''
        return self.output_states[ACTIVITY_DIFFERENCE_OUTPUT]