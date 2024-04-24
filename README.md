*Abstract*:

We extend Microsoft's cybersecurity research platform, CyberBattleSim, by integrating Multiagent Reinforcement Learning, training both the attacker and defender agent using the Q-Learning algorithm with an epsilon-greedy strategy. The interactions between these agents are modelled through a Stackelberg game, where actions are interleaved to simulate real-world cyber conflict scenarios. To accommodate the advanced capabilities of the newly trained defender agent, significant enhancements were made to the CyberBattleSim environment. The results demonstrate the effectiveness of our trained defender against the baseline CyberBattleSim defender agent, aligning with research supporting the efficacy of reinforcement learning in enhancing cybersecurity defences. Additionally, our exploration of the complexities of hyperparameters and reward model tuning in reinforcement learning algorithms reveals that adequate training significantly influences optimal agent performance in CyberBattleSim more than hyperparameter adjustments alone. We further test reward models and show a notable improvement in defender win rates against the attacker agent, demonstrating the critical role of tailored reward strategies for effective reinforcement learning defence mechanisms. We also discuss CyberBattleSim's limitations and the challenges in integrating game theory and multiagent reinforcement learning. To conclude, we propose a versatile, innovative framework for advancing reinforcement learning and game theory applications in building a comprehensive cyber battle simulation to develop robust cyber defence strategies.

*Microsoft' CyberBattleSim*
https://github.com/microsoft/CyberBattleSim?tab=readme-ov-file
https://www.microsoft.com/en-us/security/blog/2021/04/08/gamifying-machine-learning-for-stronger-security-and-ai-models/
https://github.com/microsoft/CyberBattleSim

Microsoft Open Source Code of Conduct
This project has adopted the Microsoft Open Source Code of Conduct.

*Resources*:

Microsoft Open Source Code of Conduct
Microsoft Code of Conduct FAQ
Contact opencode@microsoft.com with questions or concerns

*CyberBattleSim*

April 8th, 2021: See the announcement on the Microsoft Security Blog.

CyberBattleSim is an experimentation research platform to investigate the interaction of automated agents operating in a simulated abstract enterprise network environment. The simulation provides a high-level abstraction of computer networks and cyber security concepts. Its Python-based Open AI Gym interface allows for the training of automated agents using reinforcement learning algorithms.

The simulation environment is parameterized by a fixed network topology and a set of vulnerabilities that agents can utilize to move laterally in the network. The goal of the attacker is to take ownership of a portion of the network by exploiting vulnerabilities that are planted in the computer nodes. While the attacker attempts to spread throughout the network, a defender agent watches the network activity and tries to detect any attack taking place and mitigate the impact on the system by evicting the attacker. We provide a basic stochastic defender that detects and mitigates ongoing attacks based on pre-defined probabilities of success. We implement mitigation by re-imaging the infected nodes, a process abstractly modeled as an operation spanning over multiple simulation steps.

To compare the performance of the agents we look at two metrics: the number of simulation steps taken to attain their goal and the cumulative rewards over simulation steps across training epochs.

Project goals
We view this project as an experimentation platform to conduct research on the interaction of automated agents in abstract simulated network environments. By open-sourcing it, we hope to encourage the research community to investigate how cyber-agents interact and evolve in such network environments.

The simulation we provide is admittedly simplistic, but this has advantages. Its highly abstract nature prohibits direct application to real-world systems thus providing a safeguard against potential nefarious use of automated agents trained with it. At the same time, its simplicity allows us to focus on specific security aspects we aim to study and quickly experiment with recent machine learning and AI algorithms.

For instance, the current implementation focuses on the lateral movement cyber-attacks techniques, with the hope of understanding how network topology and configuration affects them. With this goal in mind, we felt that modeling actual network traffic was not necessary. This is just one example of a significant limitation in our system that future contributions might want to address.

On the algorithmic side, we provide some basic agents as starting points, but we would be curious to find out how state-of-the-art reinforcement learning algorithms compare to them. We found that the large action space intrinsic to any computer system is a particular challenge for Reinforcement Learning, in contrast to other applications such as video games or robot control. Training agents that can store and retrieve credentials is another challenge faced when applying RL techniques where agents typically do not feature internal memory. These are other areas of research where the simulation could be used for benchmarking purposes.

Other areas of interest include the responsible and ethical use of autonomous cyber-security systems: How to design an enterprise network that gives an intrinsic advantage to defender agents? How to conduct safe research aimed at defending enterprises against autonomous cyber-attacks while preventing nefarious use of such technology?
