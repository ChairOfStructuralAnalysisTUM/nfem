from nfem import Model, Assembler, History, plotting_utility
import numpy as np

model = Model('Initial Model')
model.AddNode(id='A', x= 0, y=0, z=0)
model.AddNode(id='B', x= 5, y=2, z=0)
model.AddNode(id='C', x=10, y=0, z=0)
model.AddTrussElement(id=1, node_a='A', node_b='B', youngs_modulus=10, area=2)
model.AddTrussElement(id=2, node_a='B', node_b='C', youngs_modulus=10, area=2)
model.AddDirichletCondition(node_id='A', dof_types='uvw', value=0)
model.AddDirichletCondition(node_id='B', dof_types='w'  , value=0)
model.AddDirichletCondition(node_id='C', dof_types='uvw', value=0)
model.AddSingleLoad(id='F', node_id='B', fv=-1)

history = History(model)
n_steps = 10
lam = 0.1

for step in range(1,n_steps):
    model.PerformLinearSolutionStep(lam*step)
    history.AddModel(step, model)

print('Initial:')
print(model.nodes['B'].x)
print(model.nodes['B'].y)
print(model.nodes['B'].z)

for step in range(1,n_steps):
    print('Deformed step',step,':')
    deformed = history.GetModel(step)
    print(deformed.nodes['B'].x)
    print(deformed.nodes['B'].y)
    print(deformed.nodes['B'].z)

plotting_utility.plot_cont_animated(history,speed=10)
