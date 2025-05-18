import torch


def get_game_time(game_clock: float, quarter: int) -> float:
    """
    Convert game clock and quarter to continuous game time in seconds.

    Args:
        game_clock: Seconds remaining in the quarter
        quarter: Quarter number (1-4 for regulation, 5+ for overtime)

    Returns:
        Total seconds elapsed since the start of the game
    """
    if quarter <= 4:
        return (quarter - 1) * 720 + (720 - game_clock)
    return 4 * 720 + (quarter - 5) * 300 + (300 - game_clock)


def split_list_by_velocity(agent_velocities: torch.Tensor):
    fragment_per_agent = []
    for agent_velocity in agent_velocities:
        fragments = []
        start_index = None

        for i, num in enumerate(agent_velocity):
            if num == 1:
                if start_index is None:
                    start_index = i
            elif start_index is not None:
                fragments.append((start_index, i - 1))
                start_index = None

        if start_index is not None:
            fragments.append((start_index, len(agent_velocity) - 1))
        fragment_per_agent.append(fragments)

    return fragment_per_agent


def team_pooling(out: torch.Tensor):
    # print(out.shape)
    player = out[:, 1:, :, :]
    ball = out[:, 0, :, :]

    player_max, _ = torch.max(player, dim=1)
    player_ball = torch.stack([player_max, ball], dim=1)
    # print(player_ball.shape)
    mean_tensor = torch.mean(player_ball, dim=[1, 2])

    return mean_tensor


def player_pooling(out: torch.Tensor):
    player = out[:, 1:, :, :]
    player_pooled = torch.mean(player, dim=[2])
    return player_pooled
