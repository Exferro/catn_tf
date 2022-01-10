from enum import Enum
# A enum to account for two possible values which nodes can take.
# 'qubit' corresponds to {0, 1}, 'spin' corresponds to {1, -1}.
# Note that switch from 'qubit' to 'spin' is 0 -> 1, 1 -> -1, and not vice versa.
Particle = Enum('Particle', 'qubit spin')
