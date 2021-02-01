
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#Antecedents/Consequents
D = ctrl.Antecedent(np.arange(0,10.5,0.5), "distance")
A = ctrl.Antecedent(np.arange(0, 91, 1), "angle")
S = ctrl.Consequent(np.arange(0, 5.2, 0.2), "speed", defuzzify_method="mom")
ST = ctrl.Consequent(np.arange(0, 91, 1), "steering", defuzzify_method="lom")

#Memberships
D["N"] = fuzz.trimf(D.universe,[0,0,5])
D["F"] = fuzz.trimf(D.universe, [2.5,5,7.5])
D["VF"] = fuzz.trimf(D.universe, [5,10,10])

A["S"] = fuzz.trimf(A.universe,[0, 0, 20])
A["M"] = fuzz.trimf(A.universe, [15, 40, 60])
A["L"] = fuzz.trimf(A.universe, [55, 90, 90])

S["SS"] = fuzz.trimf(S.universe, [0, 0, 1.6])
S["MS"] = fuzz.trimf(S.universe, [1.2,2,2.8])
S["FS"] = fuzz.trimf(S.universe, [2.4, 3.2, 4])
S["MXS"] = fuzz.trimf(S.universe, [3.6, 5, 5])

ST["MST"] = fuzz.trimf(A.universe,[0, 0, 35])
ST["SST"] = fuzz.trimf(A.universe, [25, 45, 60])
ST["VST"] = fuzz.trimf(A.universe, [55, 90, 90])

#Rules

rule1 = ctrl.Rule(D["N"] & A["S"], (S["SS"],ST["VST"]))
rule2 = ctrl.Rule(D["N"] & A["M"], (S["SS"], ST["SST"]))
rule3 = ctrl.Rule(D["N"] & A["L"], (S["MS"], ST["MST"]))
rule4 = ctrl.Rule(D["F"] & A["S"], (S["MS"], ST["VST"]))
rule5 = ctrl.Rule(D["F"] & A["M"], (S["MS"], ST["SST"]))
rule6 = ctrl.Rule(D["F"] & A["L"], (S["FS"], ST["MST"]))
rule7 = ctrl.Rule(D["VF"] & A["S"], (S["FS"], ST["VST"]))
rule8 = ctrl.Rule(D["VF"] & A["M"], (S["MXS"], ST["SST"]))
rule9 = ctrl.Rule(D["VF"] & A["L"], (S["MXS"], ST["MST"]))


#create system
action_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
action = ctrl.ControlSystemSimulation(action_ctrl)

action.input["distance"] = 8 #modify here to test different scenarios - P1
action.input["angle"] = 60 #modify here to test different scenarios - P2

'''D.view()
A.view()
S.view()
ST.view()'''

action.compute()
print(action.output)


