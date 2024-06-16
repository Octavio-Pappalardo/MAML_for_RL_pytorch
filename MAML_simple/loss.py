
from torch.func import functional_call

def policy_loss(agent,buffer,params=None):

    if params:
        _ , newlogprob, _  = functional_call(agent,params,(buffer.observations, buffer.actions) )
    else:
        _ , newlogprob, _ = agent(buffer.observations, buffer.actions)

    #normalize advantages
    buffer.advantages= (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

    pg_loss = -(newlogprob * buffer.advantages).mean()

    return pg_loss