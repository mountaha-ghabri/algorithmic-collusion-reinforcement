include("input/init.jl")
include("input/qlearning.jl")

# Init algorithm
game = init.init_game();

# Compute equilibrium
game_equilibrium = qlearning.simulate_game(game);
