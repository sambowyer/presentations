from alan import Normal, Bernoulli, Plate, BoundPlate, OptParam, Data, Problem
import torch as t

# Set up the model
d_z = 10

P = Plate(
    mu_z = Normal(t.zeros((d_z,)), t.ones((d_z,))),
    psi_z = Normal(t.zeros((d_z,)), t.ones((d_z,))),
    plate_1 = Plate(
        z = Normal("mu_z", lambda psi_z: psi_z.exp()),
        plate_2 = Plate(
            obs = Bernoulli(logits = lambda z, x: z @ x),
        )
    ),
)

Q = Plate(
    mu_z = Normal(OptParam(t.zeros((d_z,))), OptParam(t.zeros((d_z,)), transformation=t.exp)),
    psi_z = Normal(OptParam(t.zeros((d_z,))), OptParam(t.zeros((d_z,)), transformation=t.exp)),
    plate_1 = Plate(
        z = Normal(OptParam(t.zeros((d_z,))), OptParam(t.zeros((d_z,)), transformation=t.exp)),
        plate_2 = Plate(
            obs = Data()
        )
    ),
)

P = BoundPlate(P, platesizes={'plate_1': num_users, 'plate_2': num_movies}, inputs = {'x': x})
Q = BoundPlate(Q, platesizes={'plate_1': num_users, 'plate_2': num_movies}, inputs = {'x': x})

prob = Problem(P, Q)
opt = t.optim.Adam(prob.Q.parameters(), lr=lr)

# Train Q with VI
for i in range(num_iterations):
    opt.zero_grad()
    elbo = prob.sample(K=K).elbo_vi()
    elbo.backward()
    opt.step()


