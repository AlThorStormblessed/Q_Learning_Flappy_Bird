tate, verbose = 0)
        qn = model.predict(state_new, verbose = 0)

        if done:
            target = reward
        else:
            target = reward + discount * np.max(qn[0])

        q[0][action] = target
        model.train_on_batch(state, q)