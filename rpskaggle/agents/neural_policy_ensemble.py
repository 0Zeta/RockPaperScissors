from random import randint
from rpskaggle.policies import *
import tensorflow.keras as keras


logging.basicConfig(level=logging.INFO)


class NeuralPolicyEnsembleAgent(RPSAgent):
    """
    evaluates the performance of different policies and assigns each policy a weight based on the policy´s
    historical performance using a neural network
    """

    def __init__(self, configuration, strict: bool = False):
        super().__init__(configuration)
        self.strict_agent = strict
        self.policies = get_policies()
        if self.strict_agent:
            self.policies = [
                policy for policy in self.policies if policy.is_deterministic
            ]
        self.name_to_policy = {policy.name: policy for policy in self.policies}

        # Create a data frame with the historical performance of the policies
        policy_names = [policy.name for policy in self.policies]
        self.policies_performance = pd.DataFrame(columns=["step"] + policy_names)
        self.policies_performance.set_index("step", inplace=True)

        self.decay = 0.93

        # the amount of timesteps to use for predictions
        self.look_back = 6

        # simple neural network
        self.model = keras.Sequential()
        self.model.add(
            keras.layers.LSTM(
                64,
                input_shape=(self.look_back, len(self.policies)),
                return_sequences=True,
                use_bias=True,
            )
        )
        self.model.add(
            keras.layers.LSTM(
                16,
                input_shape=(self.look_back, len(self.policies)),
                return_sequences=False,
                use_bias=True,
            )
        )
        self.model.add(
            keras.layers.Dense(len(self.policies), activation="sigmoid", use_bias=True)
        )
        self.model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(learning_rate=0.005),
        )

        self.batch_size = 20

        # Data
        self.X = []
        self.y = []

    def act(self) -> int:
        if len(self.history) > 0:
            # Update the historical performance for each policy and for each decay value
            self.update_data()

        # Get the new probabilities from every policy
        policy_probs = np.array(
            [
                policy.probabilities(self.step, self.score, self.history)
                for policy in self.policies
            ]
        )
        if len(self.history) > self.look_back + self.batch_size:
            self.train_model()
        if len(self.history) > self.look_back + self.batch_size + 1:
            sample = self.policies_performance.to_numpy().astype(np.float32)[
                -self.look_back :, :
            ]
            preds = self.model.predict(sample.reshape((1, self.look_back, -1)))[0]
            preds = np.clip(preds, a_min=0, a_max=1)
            probabilities = policy_probs[np.argmax(preds)]
        else:
            probabilities = EQUAL_PROBS
        logging.info(
            "Neural Policy Ensemble | Step "
            + str(self.step)
            + " | score: "
            + str(self.score)
            + " probabilities: "
            + str(probabilities)
        )
        # Play randomly for the first 130-230 steps
        if (
            self.step < 130 + randint(0, 100)
            and get_score(self.alternate_history, 15) < 6
        ):
            action = self.random.randint(0, 2)
            if self.random.randint(0, 3) == 1:
                # We don´t want our random seed to be cracked.
                action = (action + 1) % SIGNS
            return action
        if self.strict_agent:
            action = int(np.argmax(probabilities))
        else:
            action = int(np.random.choice(range(SIGNS), p=probabilities))
        return action

    def update_data(self):
        # Determine the scores for the different actions (Win: 0.9, Tie: 0.5, Loss: 0.1)
        scores = [0, 0, 0]
        opponent_action = self.obs.lastOpponentAction
        scores[(opponent_action + 1) % 3] = 0.9
        scores[opponent_action] = 0.5
        scores[(opponent_action + 2) % 3] = 0.1

        # Policies
        for policy_name, policy in self.name_to_policy.items():
            # Calculate the policy´s score for the last step
            probs = policy.history[-1]
            score = np.sum(probs * scores)
            # Save the score to the performance data frame
            self.policies_performance.loc[self.step - 1, policy_name] = score

        if len(self.policies_performance) > self.look_back:
            # Append to X and y
            perf = self.policies_performance.to_numpy()
            self.X.append(perf[-self.look_back - 1 : -1, :])
            self.y.append(
                perf[-1, :].reshape(
                    -1,
                )
            )

    def train_model(self):
        X = np.asarray(self.X).astype(np.float32)
        y = np.asarray(self.y).astype(np.float32)

        if len(X) > 10 * self.batch_size:
            # No need to train on old data when everyone plays randomly for the first steps
            X = X[-10 * self.batch_size :]
            y = y[-10 * self.batch_size :]

        # Favor more recent samples
        weights = np.flip(np.power(self.decay, np.arange(0, len(X))))

        self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
            shuffle=True,
            sample_weight=weights,
        )


AGENT = None


def neural_policy_ensemble(observation, configuration) -> int:
    global AGENT
    if AGENT is None:
        AGENT = NeuralPolicyEnsembleAgent(configuration)
    action, history = AGENT.agent(observation)
    return action
