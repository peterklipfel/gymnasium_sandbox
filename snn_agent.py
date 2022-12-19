# SNN-related imports
import snntorch as snn
from snntorch import spikeplot as splt
import snntorch.functional as SF
from snntorch import spikegen, utils, surrogate

from agent import Agent

class SNNAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.observation_info = None

        self.beta = 0.9  # neuron decay rate
        self.spike_grad = surrogate.fast_sigmoid()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #  Initialize Network
        self.net = nn.Sequential(nn.Conv2d(1, 8, 5),
                                 nn.MaxPool2d(2),
                                 snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                                 nn.Conv2d(8, 16, 5),
                                 nn.MaxPool2d(2),
                                 snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                                 nn.Flatten(),
                                 nn.Linear(16*4*4, 10),
                                 snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                                 ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=2e-3, betas=(0.9, 0.999))
        self.loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)


    def forward_pass(self, data, num_steps):
        spk_rec = []
        utils.reset(self.net)  # resets hidden states for all LIF neurons in net

        for step in range(num_steps):
            spk_out, mem_out = self.net(data)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)

    def action(self, observation, reward=None):
        print(observation, reward)
        return 1