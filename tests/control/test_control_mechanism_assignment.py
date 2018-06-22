import pytest
import numpy as np
import psyneulink as pnl


def test_control_mechanism_assignment():
    '''ControlMechanism assignment/replacement,  monitor_for_control, and control_signal specifications'''

    T1 = pnl.TransferMechanism(size=3, name='T-1')
    T2 = pnl.TransferMechanism(function=pnl.Logistic, output_states=[{pnl.NAME: 'O-1'}], name='T-2')
    T3 = pnl.TransferMechanism(function=pnl.Logistic, name='T-3')
    T4 = pnl.TransferMechanism(function=pnl.Logistic, name='T-4')
    P = pnl.Process(pathway=[T1, T2, T3, T4])
    S = pnl.System(processes=P,
                   # controller=pnl.EVCControlMechanism,
                   controller=pnl.EVCControlMechanism(
                           control_signals=[(pnl.GAIN, T2)]
                   ),
                   enable_controller=True,
                   # Test for use of 4-item tuple with matrix in monitor_for_control specification
                   monitor_for_control=[(T1, None, None, np.ones((3,1))),
                                        ('O-1', 1, -1)],
                   control_signals=[(pnl.GAIN, T3)]
                   )
    assert len(S.controller.objective_mechanism.monitored_output_states)==2
    assert len(S.control_signals)==2

    # Test for avoiding duplicate assignment of monitored_output_states and control_signals
    C1 = pnl.EVCControlMechanism(name='C-1',
                                 objective_mechanism = [(T1, None, None, np.ones((3,1)))],
                                 control_signals=[(pnl.GAIN, T3)]
                                 )

    # Test direct assignment
    S.controller = C1
    assert len(C1.monitored_output_states)==2
    assert len(S.control_signals)==3
    assert S.controller.name == 'C-1'


    # Test for adding a monitored_output_state and control_signal
    C2 = pnl.EVCControlMechanism(name='C-2',
                                 objective_mechanism = [T3.output_states[pnl.RESULTS]],
                                 control_signals=[(pnl.GAIN, T4)])
    # Test use of assign_as_controller method
    C2.assign_as_controller(S)
    assert len(C2.monitored_output_states)==3
    assert len(S.control_signals)==4
    assert S.controller.name == 'C-2'

def test_control_mechanism_assignment_additional():
    '''Tests "free-standing" specifications of monitor_for_control and ControlSignal (i.e., outside of a list)'''
    T = pnl.TransferMechanism(name='T')
    S = pnl.sys(T,
                controller=pnl.EVCControlMechanism(),
                monitor_for_control=T,
                control_signals=(pnl.SLOPE, T),
                enable_controller=True)
    assert S.controller.objective_mechanism.input_state.path_afferents[0].sender.owner == T
    assert T.parameter_states[pnl.SLOPE].mod_afferents[0].sender.owner == S.controller

def test_prediction_mechanism_assignment():
    '''Tests prediction mechanism assignment and more tests for ObjectiveMechanism and ControlSignal assignments'''


    wf = lambda x: [x[0],x[1]]
    T = pnl.TransferMechanism(name='T')

    S = pnl.sys(T,
                controller=pnl.EVCControlMechanism(name='EVC',
                                                   prediction_mechanisms=(pnl.PredictionMechanism,
                                                                          {pnl.FUNCTION:pnl.INPUT_SEQUENCE,
                                                                           pnl.RATE:1,
                                                                           pnl.WINDOW_SIZE:3,
                                                                           # pnl.WINDOWING_FUNCTION:wf
                                                                           }),
                                                   objective_mechanism=[T]
                                                   ),
                control_signals=pnl.ControlSignal(allocation_samples=[0.1, 0.5, 0.9],
                                                   projections=(pnl.SLOPE, T)),
                enable_controller=True
                )


