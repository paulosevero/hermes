"""Contains a mobility model where users are immobile."""


def immobile(user: object):
    user.coordinates_trace.extend([user.base_station.coordinates for _ in range(5000)])
