"""Contains a custom power model that sets the device's power consumption
according to a linear interpolation between known power values.
"""


class CustomPowerModel:
    @classmethod
    def linear_interpolation(known_power_values, demand):
        interpolated_power_consumption = known_power_values[0][1] + (demand - known_power_values[0][0]) * (
            (known_power_values[1][1] - known_power_values[0][1]) / (known_power_values[1][0] - known_power_values[0][0])
        )

        return interpolated_power_consumption

    @classmethod
    def get_power_consumption(cls, device: object):
        power_consumption = CustomPowerModel.linear_interpolation(
            known_power_values=device.power_model_parameters["known_power_values"],
            demand=device.cpu_demand,
        )
        return power_consumption
