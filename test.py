from tools.history import History

history = History()
history.insert(([0, 1], 1, 2, [1, 0]))
history.insert(([0, 1], 1, 5, [2, 0]))
history.insert(([0, 1], 1, 7, [3, 0]))
# history.clear()
history.get_total_reward()
print(history.get_steps_count())

print(history.get_state_sequence())