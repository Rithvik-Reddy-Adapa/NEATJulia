#=
# Refer src/AbstractTypes.jl for Probabilities
=#

export MutationProbability, CheckMutationProbability, CrossoverProbability, CheckCrossoverProbability

@kwdef mutable struct MutationProbability <: Probabilities
  no_mutation::Real = 1.0

  # global mutations
  global_change_weight::Real = 1.0
  global_shift_weight::Real = 1.0
  global_change_bias::Real = 1.0
  global_shift_bias::Real = 1.0
  global_change_activation_function::Real = 1.0
  global_toggle_enable::Real = 1.0
  global_enable_gene::Real = 1.0
  global_disable_gene::Real = 1.0

  # input node mutations
  input_node_change_bias::Real = 0.0
  input_node_shift_bias::Real = 0.0
  input_node_change_activation_function::Real = 0.0

  # output node mutations
  output_node_change_bias::Real = 0.0
  output_node_shift_bias::Real = 0.0
  output_node_change_activation_function::Real = 0.0

  # forward connection mutations
  add_forward_connection::Real = 0.5
  forward_connection_change_weight::Real = 0.0
  forward_connection_shift_weight::Real = 0.0
  forward_connection_toggle_enable::Real = 0.0
  forward_connection_enable_gene::Real = 0.0
  forward_connection_disable_gene::Real = 0.0

  # backward connection mutations
  add_backward_connection::Real = 0.01
  backward_connection_change_weight::Real = 0.0
  backward_connection_shift_weight::Real = 0.0
  backward_connection_toggle_enable::Real = 0.0
  backward_connection_enable_gene::Real = 0.0
  backward_connection_disable_gene::Real = 0.0

  # hidden node mutations
  add_hidden_node_forward_connection::Real = 0.01
  add_hidden_node_backward_connection::Real = 0.01
  hidden_node_change_bias::Real = 0.0
  hidden_node_shift_bias::Real = 0.0
  hidden_node_change_activation_function::Real = 0.0
  hidden_node_toggle_enable::Real = 0.0
  hidden_node_enable_gene::Real = 0.0
  hidden_node_disable_gene::Real = 0.0

  # recurrent hidden node mutations
  add_recurrent_hidden_node_forward_connection::Real = 0.01
  add_recurrent_hidden_node_backward_connection::Real = 0.01
  recurrent_hidden_node_change_weight::Real = 0.0
  recurrent_hidden_node_shift_weight::Real = 0.0
  recurrent_hidden_node_change_bias::Real = 0.0
  recurrent_hidden_node_shift_bias::Real = 0.0
  recurrent_hidden_node_change_activation_function::Real = 0.0
  recurrent_hidden_node_toggle_enable::Real = 0.0
  recurrent_hidden_node_enable_gene::Real = 0.0
  recurrent_hidden_node_disable_gene::Real = 0.0

  # LSTM node mutations
  add_lstm_node_forward_connection::Real = 0.01
  add_lstm_node_backward_connection::Real = 0.01
  lstm_node_change_weight::Real = 0.0
  lstm_node_shift_weight::Real = 0.0
  lstm_node_change_bias::Real = 0.0
  lstm_node_shift_bias::Real = 0.0
  lstm_node_toggle_enable::Real = 0.0
  lstm_node_enable_gene::Real = 0.0
  lstm_node_disable_gene::Real = 0.0

  # GRU node mutations
  add_gru_node_forward_connection::Real = 0.01
  add_gru_node_backward_connection::Real = 0.01
  gru_node_change_weight::Real = 0.0
  gru_node_shift_weight::Real = 0.0
  gru_node_change_bias::Real = 0.0
  gru_node_shift_bias::Real = 0.0
  gru_node_toggle_enable::Real = 0.0
  gru_node_enable_gene::Real = 0.0
  gru_node_disable_gene::Real = 0.0
end

function CheckMutationProbability(x::MutationProbability)
  for i in fieldnames(MutationProbability)
    (getfield(x, i) < 0.0) && error("MutationProbability : got negative value for field $(i)")
    (isfinite(getfield(x, i))) || error("MutationProbability : got a non finite Real number for field $(i)")
    (getfield(x, i) > 1e100) && error("MutationProbability : value greater than 1e100 for field $(i)")
  end
  return
end

function Base.show(io::IO, x::MutationProbability)
  println(io, summary(x))
  print(io, " no_mutation : $(x.no_mutation)
 
 global_change_weight : $(x.global_change_weight)
 global_shift_weight : $(x.global_shift_weight)
 global_change_bias : $(x.global_change_bias)
 global_shift_bias : $(x.global_shift_bias)
 global_change_activation_function : $(x.global_change_activation_function)
 global_toggle_enable : $(x.global_toggle_enable)
 global_enable_gene : $(x.global_enable_gene)
 global_disable_gene : $(x.global_disable_gene)

 input_node_change_bias : $(x.input_node_change_bias)
 input_node_shift_bias : $(x.input_node_shift_bias)
 input_node_change_activation_function : $(x.input_node_change_activation_function)

 output_node_change_bias : $(x.output_node_change_bias)
 output_node_shift_bias : $(x.output_node_shift_bias)
 output_node_change_activation_function : $(x.output_node_change_activation_function)

 add_forward_connection : $(x.add_forward_connection)
 forward_connection_change_weight : $(x.forward_connection_change_weight)
 forward_connection_shift_weight : $(x.forward_connection_shift_weight)
 forward_connection_toggle_enable : $(x.forward_connection_toggle_enable)
 forward_connection_enable_gene : $(x.forward_connection_enable_gene)
 forward_connection_disable_gene : $(x.forward_connection_disable_gene)

 add_backward_connection : $(x.add_backward_connection)
 backward_connection_change_weight : $(x.backward_connection_change_weight)
 backward_connection_shift_weight : $(x.backward_connection_shift_weight)
 backward_connection_toggle_enable : $(x.backward_connection_toggle_enable)
 backward_connection_enable_gene : $(x.backward_connection_enable_gene)
 backward_connection_disable_gene : $(x.backward_connection_disable_gene)

 add_hidden_node_forward_connection : $(x.add_hidden_node_forward_connection)
 add_hidden_node_backward_connection : $(x.add_hidden_node_backward_connection)
 hidden_node_change_bias : $(x.hidden_node_change_bias)
 hidden_node_shift_bias : $(x.hidden_node_shift_bias)
 hidden_node_change_activation_function : $(x.hidden_node_change_activation_function)
 hidden_node_toggle_enable : $(x.hidden_node_toggle_enable)
 hidden_node_enable_gene : $(x.hidden_node_enable_gene)
 hidden_node_disable_gene : $(x.hidden_node_disable_gene)

 add_recurrent_hidden_node_forward_connection : $(x.add_recurrent_hidden_node_forward_connection)
 add_recurrent_hidden_node_backward_connection : $(x.add_recurrent_hidden_node_backward_connection)
 recurrent_hidden_node_change_weight : $(x.recurrent_hidden_node_change_weight)
 recurrent_hidden_node_shift_weight : $(x.recurrent_hidden_node_shift_weight)
 recurrent_hidden_node_change_bias : $(x.recurrent_hidden_node_change_bias)
 recurrent_hidden_node_shift_bias : $(x.recurrent_hidden_node_shift_bias)
 recurrent_hidden_node_change_activation_function : $(x.recurrent_hidden_node_change_activation_function)
 recurrent_hidden_node_toggle_enable : $(x.recurrent_hidden_node_toggle_enable)
 recurrent_hidden_node_enable_gene : $(x.recurrent_hidden_node_enable_gene)
 recurrent_hidden_node_disable_gene : $(x.recurrent_hidden_node_disable_gene)

 add_lstm_node_forward_connection : $(x.add_lstm_node_forward_connection)
 add_lstm_node_backward_connection : $(x.add_lstm_node_backward_connection)
 lstm_node_change_weight : $(x.lstm_node_change_weight)
 lstm_node_shift_weight : $(x.lstm_node_shift_weight)
 lstm_node_change_bias : $(x.lstm_node_change_bias)
 lstm_node_shift_bias : $(x.lstm_node_shift_bias)
 lstm_node_toggle_enable : $(x.lstm_node_toggle_enable)
 lstm_node_enable_gene : $(x.lstm_node_enable_gene)
 lstm_node_disable_gene : $(x.lstm_node_disable_gene)

 add_gru_node_forward_connection : $(x.add_gru_node_forward_connection)
 add_gru_node_backward_connection : $(x.add_gru_node_backward_connection)
 gru_node_change_weight : $(x.gru_node_change_weight)
 gru_node_shift_weight : $(x.gru_node_shift_weight)
 gru_node_change_bias : $(x.gru_node_change_bias)
 gru_node_shift_bias : $(x.gru_node_shift_bias)
 gru_node_toggle_enable : $(x.gru_node_toggle_enable)
 gru_node_enable_gene : $(x.gru_node_enable_gene)
 gru_node_disable_gene : $(x.gru_node_disable_gene)
 ")
  return
end

@kwdef mutable struct CrossoverProbability <: Probabilities
  intraspecie_good_good::Real = 1.0
  intraspecie_good_bad::Real = 1.0
  intraspecie_bad_bad::Real = 1.0

  interspecie_good_good::Real = 1.0
  interspecie_good_bad::Real = 1.0
  interspecie_bad_bad::Real = 1.0
end

function CheckCrossoverProbability(x::CrossoverProbability)
  for i in fieldnames(CrossoverProbability)
    (getfield(x, i) < 0.0) && error("CrossoverProbability : got negative value for field $(i)")
    (isfinite(getfield(x, i))) || error("CrossoverProbability : got a non finite Real number for field $(i)")
    (getfield(x, i) > 1e100) && error("CrossoverProbability : value greater than 1e100 for field $(i)")
  end
  return
end

function Base.show(io::IO, x::CrossoverProbability)
  println(io, summary(x))
  print(io, " intraspecie_good_good : $(x.intraspecie_good_good)
 intraspecie_good_bad : $(x.intraspecie_good_bad)
 intraspecie_bad_bad : $(x.intraspecie_bad_bad)

 interspecie_good_good : $(x.interspecie_good_good)
 interspecie_good_bad : $(x.interspecie_good_bad)
 interspecie_bad_bad : $(x.interspecie_bad_bad)
 ")
end



