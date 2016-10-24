# from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import *
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Projections.ControlSignal import ControlSignal
from PsyNeuLink.Functions.System import system
from PsyNeuLink.Functions.Mechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Globals.Run import run, _construct_inputs


# Preferences:
DDM_prefs = FunctionPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(False,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))

# Mechanisms:
Input = Transfer(name='Input')
Reward = Transfer(name='Reward')
Decision = DDM(function=BogaczEtAl(drift_rate=(1.0, ControlSignal(function=Linear)),
                                   # # BUG:
                                   # threshold=(1.0, ControlSignal(function=Linear)),
                                   noise=(0.5),
                                   starting_point=(0),
                                   T0=0.45),
               prefs = DDM_prefs,
               name='Decision')

# Processes:
TaskExecutionProcess = process(
    default_input_value=[0],
    pathway=[(Input, 0), IDENTITY_MATRIX, (Decision, 0)],
    prefs = process_prefs,
    name = 'TaskExecutionProcess')

RewardProcess = process(
    default_input_value=[0],
    pathway=[(Reward, 1)],
    prefs = process_prefs,
    name = 'RewardProcess')

# System:
mySystem = system(processes=[TaskExecutionProcess, RewardProcess],
                  controller=EVCMechanism,
                  enable_controller=True,
                  monitored_output_states=[Reward, PROBABILITY_UPPER_BOUND,(RT_MEAN, -1, 1)],
                  # monitored_output_states=[Reward, DECISION_VARIABLE,(RT_MEAN, -1, 1)],
                  name='EVC Test System')

# Show characteristics of system:
mySystem.show()
mySystem.controller.show()

# Specify stimuli for run:
#   two ways to do so:
#   - as a dictionary of stimulus lists; for each entry:
#     key is name of an origin mechanism in the system
#     value is a list of its sequence of stimuli (one for each trial)
inputList = [0.5, 0.123]
rewardList = [20, 20]
# stim_list_dict = {Input:[0.5, 0.123],
#               Reward:[20, 20]}
stim_list_dict = {Input:[[0.5], [0.123]],
              Reward:[[20], [20]]}
# stimDictInput = _construct_inputs(mySystem, stim_list_dict)

#   - as a list of trials;
#     each item in the list contains the stimuli for a given trial,
#     one for each origin mechanism in the system
trial_list = [[0.5, 20], [0.123, 20]]
# trialListInput = _construct_inputs(mySystem, trial_list)
reversed_trial_list = [[Reward, Input], [20, 0.5], [20, 0.123]]
# trialListInput = _construct_inputs(mySystem, reversed_trial_list)

# Create printouts function (to call in run):
def show_trial_header():
    print("\n############################ TRIAL {} ############################".format(CentralClock.trial))

def show_results():
    results = sorted(zip(mySystem.terminalMechanisms.outputStateNames, mySystem.terminalMechanisms.outputStateValues))
    print('\nRESULTS (time step {}): '.format(CentralClock.time_step))
    print ('\tControl signal (from EVC): {}'.format(Decision.parameterStates[DRIFT_RATE].value))
    for result in results:
        print("\t{}: {}".format(result[0], result[1]))

# Run system:
# run(mySystem,
#     inputs=trialListInput,
#     call_before_trial=show_trial_header,
#     call_after_time_step=show_results
#     )
# mySystem.run(inputs=trialListInput,
#              call_before_trial=show_trial_header,
#              call_after_time_step=show_results
#              )
# mySystem.run(inputs=stim_list_dict,
#              call_before_trial=show_trial_header,
#              call_after_time_step=show_results
#              )
run(mySystem, inputs=trial_list,
             call_before_trial=show_trial_header,
             call_after_time_step=show_results
             )
# mySystem.run(inputs=stim_list_dict,
#              call_before_trial=show_trial_header,
#              call_after_time_step=show_results
#              )
