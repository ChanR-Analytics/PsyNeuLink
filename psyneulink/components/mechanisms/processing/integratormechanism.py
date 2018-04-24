# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  IntegratorMechanism *************************************************

"""
Overview
--------

An IntegratorMechanism integrates its input, possibly based on its prior values.  The input can be a single
scalar value or an array of scalars (list or 1d np.array).  If it is a list or array, then each value is
independently integrated.  The default function (`Integrator`) can be parametrized to implement either a simple
increment rate, additive accumulator, or an (exponentially weighted) time-averaging of its input.  It can also be
assigned a custom function.

.. _IntegratorMechanism_Creation:

Creating an IntegratorMechanism
-------------------------------

An IntegratorMechanism can be created directly by calling its constructor, or using the `mechanism` command and
specifying *INTEGRATOR_MECHANISM* as its **mech_spec** argument.  Its function is specified in the **function**
argument, which can be parametrized by calling its constructor with parameter values::

    >>> import psyneulink as pnl
    >>> my_time_averaging_mechanism = pnl.IntegratorMechanism(function=pnl.AdaptiveIntegrator(rate=0.5))

The **default_variable** argument specifies the format of its input (i.e., whether it is a single scalar or an
array), as well as the value to use if none is provided when Mechanism is executed.  Alternatively, the **size**
argument can be used to specify the length of the array, in which case it will be initialized with all zeros.

.. _IntegratorMechanism_Structure

Structure
---------

An IntegratorMechanism has a single `InputState`, the `value <InputState.InputState.value>` of which is
used as the  `variable <IntegratorMechanism.variable>` for its `function <IntegratorMechanism.function>`.
The default for `function <IntegratorMechanism.function>` is `AdaptiveIntegrator(rate=0.5)`. However,
a custom function can also be specified,  so long as it takes a numeric value, or a list or np.ndarray of numeric
values as its input, and returns a value of the same type and format.  The Mechanism has a single `OutputState`,
the `value <OutputState.OutputState.value>` of which is assigned the result of  the call to the Mechanism's
`function  <IntegratorMechanism.function>`.

.. _IntegratorMechanism_Execution

Execution
---------

When an IntegratorMechanism is executed, it carries out the specified integration, and assigns the
result to the `value <IntegratorMechanism.value>` of its `primary OutputState <OutputState_Primary>`.  For the default
function (`Integrator`), if the value specified for **default_variable** is a list or array, or **size** is greater
than 1, each element of the array is independently integrated.  If its `rate <Integrator.rate>` parameter is a
single value,  that rate will be used for integrating each element.  If the `rate <Integrator.rate>` parameter is a
list or array, then each element will be used as the rate for the corresponding element of the input (in this case,
`rate <Integrator.rate>` must be the same length as the value specified for **default_variable** or **size**).


.. _IntegratorMechanism_Class_Reference:

Class Reference
---------------

"""
from collections import Iterable

import typecheck as tc

from psyneulink.components.functions.function import AdaptiveIntegrator
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.globals.keywords import CLASS_DEFAULTS, INTEGRATOR_MECHANISM, RESULTS, kwPreferenceSetName
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

import functools
import ctypes
import psyneulink.llvm as pnlvm
from llvmlite import ir

__all__ = [
    'DEFAULT_RATE', 'IntegratorMechanism', 'IntegratorMechanismError'
]

# IntegratorMechanism parameter keywords:
DEFAULT_RATE = 0.5

class IntegratorMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class IntegratorMechanism(ProcessingMechanism_Base):
    """
    IntegratorMechanism(                            \
    default_variable=None,                               \
    size=None,                                              \
    function=AdaptiveIntegrator(rate=0.5), \
    params=None,                                            \
    name=None,                                              \
    prefs=None)

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that integrates its input.

    COMMENT:
        Description:
            - DOCUMENT:

        Class attributes:
            + componentType (str): SigmoidLayer
            + classPreference (PreferenceSet): SigmoidLayer_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
            + ClassDefaults.variable (value):  SigmoidLayer_DEFAULT_BIAS
            + paramClassDefaults (dict): {FUNCTION_PARAMS:{kwSigmoidLayer_Unitst: kwSigmoidLayer_NetInput
                                                                     kwSigmoidLayer_Gain: SigmoidLayer_DEFAULT_GAIN
                                                                     kwSigmoidLayer_Bias: SigmoidLayer_DEFAULT_BIAS}}
        Class methods:
            None

        MechanismRegistry:
           All instances of SigmoidLayer are registered in MechanismRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT

    Arguments
    ---------

    default_variable : number, list or np.ndarray
        the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` methods;
        also serves as a template to specify the length of `variable <IntegratorMechanism.variable>` for
        `function <IntegratorMechanism.function>`, and the `primary outputState <OutputState_Primary>` of the
        Mechanism.

    size : int, list or np.ndarray of ints
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    function : IntegratorFunction : default Integrator
        specifies the function used to integrate the input.  Must take a single numeric value, or a list or np.array
        of values, and return one of the same form.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, parameters for its `function <IntegratorMechanism.function>`, and/or a custom function and its
        parameters.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    name : str : default see `name <IntegratorMechanism.name>`
        specifies the name of the IntegratorMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the IntegratorMechanism; see `prefs <IntegratorMechanism.prefs>` for details.

    Attributes
    ----------
    variable : value: default
        the input to Mechanism's ``function``.

    name : str
        the name of the IntegratorMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the IntegratorMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    componentType = INTEGRATOR_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'IntegratorMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    class ClassDefaults(ProcessingMechanism_Base.ClassDefaults):
        # setting owner to a string to avoid using this exact instance
        # in the future, best to make this a sort of spec that can be used to
        # construct a default instance
        function = AdaptiveIntegrator(rate=0.5, owner=CLASS_DEFAULTS)

    paramClassDefaults = ProcessingMechanism_Base.paramClassDefaults.copy()
    # paramClassDefaults.update({
    #     OUTPUT_STATES:[PREDICTION_MECHANISM_OUTPUT]
    # })

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(list, dict))=None,
                 function=None,
                 output_states:tc.optional(tc.any(str, Iterable))=RESULTS,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(input_states=input_states,
                                                  output_states=output_states,
                                                  function=function,
                                                  params=params)

        # if default_variable is NotImplemented:
        #     default_variable = SigmoidLayer_DEFAULT_NET_INPUT

        # self.size = size

        super(IntegratorMechanism, self).__init__(default_variable=default_variable,
                                                  size=size,
                                                  # input_states=input_states,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  context=self,
                                                  function=function,
                                                  )

        # IMPLEMENT: INITIALIZE LOG ENTRIES, NOW THAT ALL PARTS OF THE MECHANISM HAVE BEEN INSTANTIATED

    # MODIFIED 6/2/17 NEW:
    @property
    def previous_value(self):
        return self.function_object.previous_value
    # MODIFIED 6/2/17 END

    def _gen_llvm_function_body(self, ctx, builder):
        params, context, si, so = builder.function.args

        main_function = ctx.get_llvm_function(self.function_object.llvmSymbolName)
        # Allocate temporary storage. We rely on the fact that series
        # of input state results should match the main function input.
        is_output = builder.alloca(main_function.args[2].type.pointee, 1)

        for i, state in enumerate(self.input_states):
            is_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(i)])
            is_context = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(i)])
            is_in = builder.gep(si, [ctx.int32_ty(0), ctx.int32_ty(i)])
            is_out = builder.gep(is_output, [ctx.int32_ty(0), ctx.int32_ty(i)])
            is_function = ctx.get_llvm_function(state.llvmSymbolName)
            builder.call(is_function, [is_params, is_context, is_in, is_out])

        mf_in = is_output
        mf_out = builder.alloca(main_function.args[3].type.pointee, 1)
        mf_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1)])
        mf_context = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(1)])

        builder.call(main_function, [mf_params, mf_context, mf_in, mf_out])

        for i, state in enumerate(self.output_states):
            os_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(2), ctx.int32_ty(i)])
            os_context = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(2), ctx.int32_ty(i)])
            os_output = builder.gep(so, [ctx.int32_ty(0), ctx.int32_ty(i)])
            os_input = builder.gep(mf_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
            os_function = ctx.get_llvm_function(state.llvmSymbolName)
            builder.call(os_function, [os_params, os_context, os_input, os_output])

        return builder
