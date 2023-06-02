# EdgeSimPy components
from edge_sim_py import *

# EdgeSimPy extensions
from simulator.simulator_extensions import *

# Helper methods
from simulator.helper_methods import *

# Python libraries
from random import seed


# Defining a seed value to enable reproducibility
seed(1)

# Creating list of map coordinates
map_coordinates = hexagonal_grid(x_size=4, y_size=3)

edge_server_specifications = [
    {
        "model_name": "Server 1 - Dell PowerEdge R620",
        "patch_time": 10,
        "mips": 1450,
        "cpu": 8,
        "memory": 8,
        "disk": 200,
        "static_power_percentage": 54.1 / 243,
        "max_power_consumption": 243,
        "known_power_values": [
            (0, 54.1),
            (10, 78.4),
            (20, 88.5),
            (30, 99.5),
            (40, 115),
            (50, 126),
            (60, 143),
            (70, 165),
            (80, 196),
            (90, 226),
            (100, 243),
        ],
    },
    {
        "model_name": "Server 2 - Acer AR585 F1",
        "patch_time": 10,
        "mips": 3500,
        "cpu": 10,
        "memory": 10,
        "disk": 200,
        "static_power_percentage": 127 / 559,
        "max_power_consumption": 559,
        "known_power_values": [
            (0, 127),
            (10, 220),
            (20, 254),
            (30, 293),
            (40, 339),
            (50, 386),
            (60, 428),
            (70, 463),
            (80, 497),
            (90, 530),
            (100, 559),
        ],
    },
    {
        "model_name": "Server 3 - SGI Rackable C2112-4G10",
        "patch_time": 10,
        "mips": 2750,
        "cpu": 12,
        "memory": 12,
        "disk": 200,
        "static_power_percentage": 265 / 1387,
        "max_power_consumption": 1387,
        "known_power_values": [
            (0, 265),
            (10, 531),
            (20, 624),
            (30, 718),
            (40, 825),
            (50, 943),
            (60, 1060),
            (70, 1158),
            (80, 1239),
            (90, 1316),
            (100, 1387),
        ],
    },
]

# Creating base stations for providing wireless connectivity to users,
# network switches for wired connectivity, and edge servers for hosting applications
edge_servers_per_spec = [1, 0, 2, -1, -1, 1, -1, 1, 1, 2, -1, 2]
for coordinates_id, coordinates in enumerate(map_coordinates):
    # Creating a base station object
    base_station = BaseStation()
    base_station.wireless_delay = 0
    base_station.coordinates = coordinates

    # Creating a network switch object using the "sample_switch()" generator, which embeds built-in power consumption specs
    network_switch = sample_switch()
    base_station._connect_to_network_switch(network_switch=network_switch)

    # Creating an edge server
    if edge_servers_per_spec[coordinates_id] != -1:
        spec = edge_server_specifications[edge_servers_per_spec[coordinates_id]]
        edge_server = EdgeServer()
        edge_server.model_name = spec["model_name"]

        # Computational capacity (CPU in cores, RAM memory in megabytes, disk in megabytes, and millions of instructions per second)
        edge_server.cpu = spec["cpu"]
        edge_server.memory = spec["memory"]
        edge_server.disk = spec["disk"]
        edge_server.mips = spec["mips"]
        edge_server.patch_time = spec["patch_time"]
        edge_server.status = "outdated"
        edge_server.patching_log = {
            "started_at": None,
            "finished_at": None,
        }

        # Power-related attributes
        edge_server.power_model = LinearServerPowerModel
        edge_server.power_model_parameters = {
            "known_power_values": spec["known_power_values"],
            "static_power_percentage": spec["static_power_percentage"],
            "max_power_consumption": spec["max_power_consumption"],
        }

        # Connecting the edge server to a base station
        base_station._connect_to_edge_server(edge_server=edge_server)


# Creating a partially-connected mesh network topology
partially_connected_hexagonal_mesh(
    network_nodes=NetworkSwitch.all(),
    link_specifications=[
        {"number_of_objects": 23, "delay": 1, "bandwidth": 1},
    ],
)


# Defining specifications for container images and container registries
container_image_specifications = [
    {
        "name": "image1",
        "tag": "latest",
        "digest": "image1",
        "layers": [
            {
                "digest": "aaaa",
                "size": 10,
            },
            {
                "digest": "cccc",
                "size": 10,
            },
        ],
        "layers_digests": [
            "aaaa",
            "cccc",
        ],
    },
    {
        "name": "image2",
        "tag": "latest",
        "digest": "image2",
        "layers": [
            {
                "digest": "bbbb",
                "size": 10,
            },
            {
                "digest": "cccc",
                "size": 10,
            },
        ],
        "layers_digests": [
            "bbbb",
            "cccc",
        ],
    },
    {
        "name": "image3",
        "tag": "latest",
        "digest": "image3",
        "layers": [
            {
                "digest": "dddd",
                "size": 10,
            },
            {
                "digest": "eeee",
                "size": 10,
            },
            {
                "digest": "ffff",
                "size": 10,
            },
            {
                "digest": "cccc",
                "size": 10,
            },
        ],
        "layers_digests": [
            "dddd",
            "eeee",
            "ffff",
            "cccc",
        ],
    },
    {
        "name": "registry",
        "tag": "latest",
        "digest": "registry",
        "layers": [
            {
                "digest": "yyyy",
                "size": 10,
            },
            {
                "digest": "zzzz",
                "size": 10,
            },
        ],
        "layers_digests": [
            "yyyy",
            "zzzz",
        ],
    },
]
container_registry_specifications = [
    {
        "number_of_objects": 1,
        "cpu_demand": 10,
        "memory_demand": 10,
        "images": [
            {"name": "registry", "tag": "latest"},
            {"name": "image1", "tag": "latest"},
            {"name": "image2", "tag": "latest"},
            {"name": "image3", "tag": "latest"},
        ],
    }
]

# Parsing the specifications for container images and container registries
container_registries = create_container_registries(
    container_registry_specifications=container_registry_specifications,
    container_image_specifications=container_image_specifications,
)

# Defining the initial placement for container images and container registries
provision_container_registry(container_registry_specification=container_registries[0], server=EdgeServer.find_by_id(4))

for edge_server in EdgeServer.all():
    if len(edge_server.container_registries) > 0:
        edge_server.status = "updated"
        edge_server.patching_log = {
            "started_at": 0,
            "finished_at": 0,
        }

# Creating applications and users
application_specifications = [
    {
        "id": 1,
        "delay_sla": 1,
        "services": [
            {
                "cpu_demand": 6,
                "memory_demand": 6,
                "state": 5,
                "image": "image1",
            }
        ],
    },
    {
        "id": 2,
        "delay_sla": 1,
        "services": [
            {
                "cpu_demand": 4,
                "memory_demand": 4,
                "state": 0,
                "image": "image1",
            }
        ],
    },
    {
        "id": 3,
        "delay_sla": 1,
        "services": [
            {
                "cpu_demand": 3,
                "memory_demand": 3,
                "state": 0,
                "image": "image1",
            }
        ],
    },
    {
        "id": 4,
        "delay_sla": 2,
        "services": [
            {
                "cpu_demand": 6,
                "memory_demand": 6,
                "state": 0,
                "image": "image3",
            }
        ],
    },
    {
        "id": 5,
        "delay_sla": 2,
        "services": [
            {
                "cpu_demand": 1,
                "memory_demand": 1,
                "state": 0,
                "image": "image2",
            }
        ],
    },
    {
        "id": 6,
        "delay_sla": 2,
        "services": [
            {
                "cpu_demand": 2,
                "memory_demand": 2,
                "state": 0,
                "image": "image2",
            }
        ],
    },
]

user_base_stations = [
    BaseStation.find_by_id(5).coordinates,
    BaseStation.find_by_id(5).coordinates,
    BaseStation.find_by_id(7).coordinates,
    BaseStation.find_by_id(6).coordinates,
    BaseStation.find_by_id(4).coordinates,
    BaseStation.find_by_id(8).coordinates,
]
for app_spec in application_specifications:
    # Creating the application object
    app = Application()

    # Creating the user that access the application
    user = User()

    user.communication_paths[str(app.id)] = []
    user.delays[str(app.id)] = None
    user.delay_slas[str(app.id)] = app_spec["delay_sla"]

    # Defining user's coordinates and connecting him to a base station
    user.mobility_model = immobile
    user._set_initial_position(coordinates=user_base_stations[user.id - 1], number_of_replicates=10)

    # Defining user's access pattern
    CircularDurationAndIntervalAccessPattern(
        user=user,
        app=app,
        start=1,
        duration_values=[float("inf")],
        interval_values=[0],
    )
    # Defining the relationship attributes between the user and the application
    user.applications.append(app)
    app.users.append(user)

    # Creating services
    for service_spec in app_spec["services"]:
        # Gathering information on the service image based on the specified 'name' parameter
        service_image = next((img for img in ContainerImage.all() if img.name == service_spec["image"]), None)

        # Creating the service object
        service = Service(
            image_digest=service_image.digest,
            cpu_demand=service_spec["cpu_demand"],
            memory_demand=service_spec["memory_demand"],
            state=service_spec["state"],
        )

        # Connecting the application to its new service
        app.connect_to_service(service)


# Defining the inicial service placement
predefined_placement = {
    "1": EdgeServer.find_by_id(1),
    "2": EdgeServer.find_by_id(1),
    "3": EdgeServer.find_by_id(8),
    "4": EdgeServer.find_by_id(8),
    "5": EdgeServer.find_by_id(8),
    "6": EdgeServer.find_by_id(8),
}
for service in Service.all():
    if str(service.id) in predefined_placement:
        # Gathering the service's predefined host
        edge_server = predefined_placement[str(service.id)]

        # Updating the host's resource usage
        edge_server.cpu_demand += service.cpu_demand
        edge_server.memory_demand += service.memory_demand

        # Creating relationship between the host and the registry
        service.server = edge_server
        edge_server.services.append(service)

        service._available = True
        service.being_provisioned = False

        for layer_metadata in edge_server._get_uncached_layers(service=service):
            layer = ContainerLayer(
                digest=layer_metadata.digest,
                size=layer_metadata.size,
                instruction=layer_metadata.instruction,
            )

            # Updating host's resource usage based on the layer size
            edge_server.disk_demand += layer.size

            # Creating relationship between the host and the layer
            layer.server = edge_server
            edge_server.container_layers.append(layer)

# Calculating user communication paths and application delays
for user in User.all():
    for application in user.applications:
        user.set_communication_path(app=application)

container_layers_placement = [
    {
        "edge_server": EdgeServer.find_by_id(3),
        "layers": [
            "bbbb",
        ],
    },
    {
        "edge_server": EdgeServer.find_by_id(7),
        "layers": [
            "aaaa",
        ],
    },
]
# Manually adding container layers to unpopulated edge servers
for edge_server_metadata in container_layers_placement:
    edge_server = edge_server_metadata["edge_server"]
    layers = edge_server_metadata["layers"]

    edge_server_layer_digests = [layer.digest for layer in edge_server.container_layers]

    for layer_digest in layers:
        existing_layer = ContainerLayer.find_by(attribute_name="digest", attribute_value=layer_digest)
        if layer_digest not in edge_server_layer_digests:
            # Creating the new container layer object
            layer = ContainerLayer(
                digest=existing_layer.digest,
                size=existing_layer.size,
                instruction=existing_layer.instruction,
            )
            # Updating host's resource usage based on the layer size
            edge_server.disk_demand += layer.size

            # Creating relationship between the host and the layer
            layer.server = edge_server
            edge_server.container_layers.append(layer)


print("=== APPLICATIONS ===")
for app in Application.all():
    app_metadata = {
        "user": app.users[0],
        "sla": app.users[0].delay_slas[str(app.id)],
        "delay": app.users[0].delays[str(app.id)],
        "comm_path": app.users[0].communication_paths[str(app.id)],
        "services": [f"{service.cpu_demand} ({service.image_digest})" for service in app.services],
    }
    print(f"{app_metadata}")
print("=== EDGE SERVERS ===")
for edge_server in EdgeServer.all():
    edge_server_metadata = {
        "edge_server": edge_server,
        "mips": edge_server.mips,
        "capacity": [edge_server.cpu, edge_server.memory, edge_server.disk],
        "demand": [edge_server.cpu_demand, edge_server.memory_demand, edge_server.disk_demand],
        "services": [service.id for service in edge_server.services],
        "patch_time": edge_server.patch_time,
        "status": edge_server.status,
        "patching_log": edge_server.patching_log,
    }
    print(f"{edge_server_metadata}")

# Overriding the object exporting structure to allow EdgeSimPy to import maintenance-related attributes
EdgeServer._to_dict = edge_server_to_dict
EdgeServer.collect = edge_server_collect
EdgeServer.update = edge_server_update
EdgeServer.step = edge_server_step

# Exporting scenario
ComponentManager.export_scenario(save_to_file=True, file_name="sample_dataset")
display_topology(Topology.first())
