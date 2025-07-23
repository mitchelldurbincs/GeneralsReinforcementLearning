### GameService
* CreateGame
	* CreateGameRequest
	* CreateGameResponse
* JoinGame
	* JoinGameRequest
	* JoinGameResponse
* SubmitAction
	* SubmitActionRequest
		* string game_id
		* string player_id
		* Action action
		* int32 expected_turn_number
		* optional string idempotency_key = 5;
	* SubmitActionResponse
		* GameState next_game_state
		* float reward 
		* bool done
		* optional ActionError action_error
		* google.protobuf.Timestamp processed_at
* GetReplay
	* GetReplayRequest
		* string game_id
	* GetReplayResponse
		* string game_id
		* GameState initial_state
* Entities 
	* GameState
		* string game_id
		* int32 turn_count
		* map<string, PlayerState> player_states
		* bytes map_data
		* bool is_terminal
		* optional string winner_player_id
		* optional string current_player_turn_id
		* GameStatus game_status (enum: WAITING FOR PLAYERS, IN PROGRESS, FINISHED, ABORTED)
	* PlayerState
		* string player_id
		* int32 total_army
		* int32 total_land
		* int32 general_tile_index
		* bool is_defeated
		* bytes visible_map_data
		* PlayerStatus status (enum: ACTIVE, DEFEATED, SURRENDERED)
	* Action
		* int32 from_index
		* int32 to_index
		* bool is_half_split

### MatchMaker Service
* FindMatch
	* FindMatchRequest
		* string player_id
		* other tbd
	* FindMatchResponse
* FindTournamentMatch
	* FindTournamentMatchRequest
	* FindTournamentMatchResponse

### Replay Service
* RecordExperience
	* RecordExperienceRequest
		* repeated Experience experiences
		* optional string idempotency_key = 5;
	* RecordExperienceResponse
	* Entity
		* Experience
			* bytes state
			* bytes action_taken
			* float reward_received
			* bytes next_state
			* bool done_flag
			* string agent_version_id
			* float initial_td_error_priority
			* optional string episode_id
			* optional int32 step_in_episode
* GetExperienceBatch
	* GetExperienceBatchRequest
		* int32 batch_size
		* optional SamplingStrategy strategy (enum: UNIFORM, PRIORITY_BASED)
		* optional string learner_id
	* GetExperienceBatchResponse
		* repeated Experience experiences
		* optional bytes batch_metadata (importance sampling weights if using PER)

### Model Service
* GetPolicy
	* GetPolicyRequest
		* optional string model_id_requested
		* string requesting_actor_id
		* optional map<string, string> actor_capabilities (?)
	* GetPolicyResponse
		* string model_id (actual id of the model provided)
		* bytes model_weights (serialized model weights)
		* ModelMetadata metadata
			* string architecture_name
			* int32 observation_shape
			* int32 actoin_space_size
			* google.protobuf.Timestamp trained_at
			* map <string, string> training_hyperparameters
* PublishPolicy
	* PublishPolicyRequest
		* bytes model_weights
		* ModelMetadata metadata
		* string publisher_learner_id
		* optional PerformanceMetrics evaluation_metrics
	* PublishPolicyResponse
		* string model_id_assigned
		* PublishStatus status (enum: ACCEPTED, PENDING VALIDATION, REJECTED)
		* optional string message


### Elo Service - prob not actually, might be apart of MatchMaker Service
* RegisterPolicy 
	* RegisterPolicyRequest
	* RegisterPolicyResponse
* GetEloRating
	* GetEloRatingRequest
	* GetEloRatingRespones
* UpdateEloRatings
	* UpdateEloRatingsRequest
	* UpdateEloRatingsResponse
* GetRankedPolicies
	* GetRankedPoliciesRequest
	* GetRankedPoliciesResponse