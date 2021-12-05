	## Agent	
	agent_args = {}
	
	# agent type
	agent_args['agent_type'] = 'advantage'
	
	# preprocessing
	agent_args['preprocess_input_images'] = lambda x: x / 255. - 0.5
	agent_args['preprocess_input_measurements'] = lambda x: x / 100. - 0.5
	targ_scale_coeffs = np.expand_dims((np.expand_dims(np.array([30.]),1) * np.ones((1,len(target_maker_args['future_steps'])))).flatten(),0)
	agent_args['preprocess_input_targets'] = lambda x: x / targ_scale_coeffs
	agent_args['postprocess_predictions'] = lambda x: x * targ_scale_coeffs