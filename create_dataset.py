# EdgeSimPy components
from edge_sim_py import *

# EdgeSimPy extensions
from simulator.simulator_extensions import *

# Helper methods
from simulator.helper_methods import *

# Python libraries
from random import seed, sample
import json


# Defining a seed value to enable reproducibility
seed(1)

# Creating list of map coordinates
map_coordinates = hexagonal_grid(x_size=10, y_size=10)

SERVERS_PER_SPEC = 10
SERVER_PATCH_TIME = 180
edge_server_specifications = [
    {
        "number_of_objects": SERVERS_PER_SPEC,
        "model_name": "Server 1 - Dell PowerEdge R620",
        "patch_time": SERVER_PATCH_TIME,
        "mips": 1450,
        "cpu": 16,
        "memory": 24,
        "disk": 131072,  # 128 GB
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
        "number_of_objects": SERVERS_PER_SPEC,
        "model_name": "Server 2 - Acer AR585 F1",
        "patch_time": SERVER_PATCH_TIME,
        "mips": 3500,
        "cpu": 48,
        "memory": 64,
        "disk": 131072,  # 128 GB
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
        "number_of_objects": SERVERS_PER_SPEC,
        "model_name": "Server 3 - SGI Rackable C2112-4G10",
        "patch_time": SERVER_PATCH_TIME,
        "mips": 2750,
        "cpu": 32,
        "memory": 32,
        "disk": 131072,  # 128 GB
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

# Creating base stations for providing wireless connectivity to users and network switches for wired connectivity
for coordinates_id, coordinates in enumerate(map_coordinates):
    # Creating a base station object
    base_station = BaseStation()
    base_station.wireless_delay = 0
    base_station.coordinates = coordinates

    # Creating a network switch object using the "sample_switch()" generator, which embeds built-in power consumption specs
    network_switch = sample_switch()
    base_station._connect_to_network_switch(network_switch=network_switch)

# Creating a partially-connected mesh network topology
partially_connected_hexagonal_mesh(
    network_nodes=NetworkSwitch.all(),
    link_specifications=[
        {"number_of_objects": 261, "delay": 3, "bandwidth": 12.5},
    ],
)


# Creating edge servers
for spec in edge_server_specifications:
    for _ in range(spec["number_of_objects"]):
        # Creating an edge server
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

        # Connecting the edge server to a random base station that has no edge server connected to it yet
        base_station = sample([base_station for base_station in BaseStation.all() if len(base_station.edge_servers) == 0], 1)[0]
        base_station._connect_to_edge_server(edge_server=edge_server)


# Manually creating an edge server that will be fully occupied by the container registry
CONTAINER_REGISTRY_BASE_STATION = 55
registry_host_spec = edge_server_specifications[0]
edge_server = EdgeServer()
edge_server.model_name = registry_host_spec["model_name"]

# Computational capacity (CPU in cores, RAM memory in megabytes, disk in megabytes, and millions of instructions per second)
edge_server.cpu = registry_host_spec["cpu"]
edge_server.memory = registry_host_spec["memory"]
edge_server.disk = registry_host_spec["disk"]
edge_server.mips = registry_host_spec["mips"]
edge_server.patch_time = registry_host_spec["patch_time"]
edge_server.status = "updated"
edge_server.patching_log = {
    "started_at": 0,
    "finished_at": 0,
}

# Power-related attributes
edge_server.power_model = LinearServerPowerModel
edge_server.power_model_parameters = {
    "known_power_values": registry_host_spec["known_power_values"],
    "static_power_percentage": registry_host_spec["static_power_percentage"],
    "max_power_consumption": registry_host_spec["max_power_consumption"],
}

# Connecting the edge server to a random base station that has no edge server connected to it yet
base_station = BaseStation.find_by_id(CONTAINER_REGISTRY_BASE_STATION)
base_station._connect_to_edge_server(edge_server=edge_server)


# Reading specifications for container images and container registries
with open("container_images.json", "r", encoding="UTF-8") as read_file:
    container_image_specifications = json.load(read_file)

# Manually including a "registry" image specification that is used by container registries within the infrastructure
container_registry_image = {
    "name": "registry",
    "digest": "sha256:f4d532d482a050a3bb02886be6d6deda9c22cf8df44b1465f04c8648ee573a70",
    "layers": [
        {
            "digest": "sha256:8a49fdb3b6a5ff2bd8ec6a86c05b2922a0f7454579ecc07637e94dfd1d0639b6",
            "size": 3.2400989532470703,
        },
        {
            "digest": "sha256:4cb4a93be51cb152747162cdf555b5b3bbd25dd82ff49f0304e9d00bae094e1b",
            "size": 5.634395599365234,
        },
    ],
}
container_image_specifications.append(container_registry_image)

# Adding a "latest" tag to all container images
condensed_images_metadata = []
for container_image in container_image_specifications:
    container_image["tag"] = "latest"
    condensed_images_metadata.append(
        {
            "name": container_image["name"],
            "tag": container_image["tag"],
        }
    )

container_registry_specifications = [
    {
        "number_of_objects": 1,
        "cpu_demand": registry_host_spec["cpu"],
        "memory_demand": registry_host_spec["memory"],
        "images": condensed_images_metadata,
    }
]

# Parsing the specifications for container images and container registries
container_registries = create_container_registries(
    container_registry_specifications=container_registry_specifications,
    container_image_specifications=container_image_specifications,
)

# Defining the initial placement for container images and container registries
container_registry_host = [
    edge_server for edge_server in EdgeServer.all() if edge_server.base_station.id == CONTAINER_REGISTRY_BASE_STATION
][0]
provision_container_registry(container_registry_specification=container_registries[0], server=container_registry_host)

# 3GPP. “5G; Service requirements for the 5G system (3GPP TS 22.261 version 16.16.0 Release 16)”,
# Technical specification (ts), 3rd Generation Partnership Project (3GPP), 2022, 72p.
delay_slas = [15, 20, 30]
service_demands = [
    {"cpu_demand": 2, "memory_demand": 2},
    {"cpu_demand": 4, "memory_demand": 4},
    {"cpu_demand": 8, "memory_demand": 8},
]

# Defining service image specifications
service_image_specifications = [
    {"state": 0, "image_name": "centos"},
    {"state": 0, "image_name": "ros"},
    {"state": 0, "image_name": "debian"},
    {"state": 0, "image_name": "ruby"},
    {"state": 0, "image_name": "rust"},
    {"state": 0, "image_name": "python"},
    {"state": 0, "image_name": "erlang"},
    {"state": 0, "image_name": "elixir"},
    {"state": 250, "image_name": "telegraf"},
    {"state": 0, "image_name": "storm"},
    {"state": 0, "image_name": "node"},
    {"state": 0, "image_name": "tomcat"},
    {"state": 0, "image_name": "nginx"},
    {"state": 250, "image_name": "redis"},
    {"state": 250, "image_name": "mongo"},
]
SERVICES_PER_SPEC = 6
TOTAL_SERVICES = len(service_image_specifications) * SERVICES_PER_SPEC
service_image_specification_values = uniform(n_items=TOTAL_SERVICES, valid_values=service_image_specifications)

# Creating service and user objects
for service_spec in service_image_specification_values:
    # Creating the application object
    app = Application()

    # Creating the user that access the application
    user = User()

    user.communication_paths[str(app.id)] = []
    user.delays[str(app.id)] = None

    # Defining user's coordinates and connecting him to a base station
    user.mobility_model = immobile
    random_base_station = sample(BaseStation.all(), 1)[0]
    user._set_initial_position(coordinates=random_base_station.coordinates, number_of_replicates=2)

    # Defining the user's access pattern
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

    # Gathering information on the service image based on the specified 'name' parameter
    service_image = next((img for img in ContainerImage.all() if img.name == service_spec["image_name"]), None)

    # Creating the service object
    service = Service(
        image_digest=service_image.digest,
        cpu_demand=0,  # This attribute will be properly defined later
        memory_demand=0,  # This attribute will be properly defined later
        state=service_spec["state"],
    )

    # Connecting the application to its new service
    app.connect_to_service(service)

# Defining application delay SLAs
applications_sorted_randomly = sample(Application.all(), Application.count())
application_sla_values = uniform(n_items=Application.count(), valid_values=delay_slas)
for index, application in enumerate(applications_sorted_randomly):
    application.users[0].delay_slas[str(application.id)] = application_sla_values[index]

# Defining service demands
services_sorted_randomly = sample(Service.all(), Service.count())
service_demand_values = uniform(n_items=Service.count(), valid_values=service_demands)
for index, service in enumerate(services_sorted_randomly):
    service.cpu_demand = service_demand_values[index]["cpu_demand"]
    service.memory_demand = service_demand_values[index]["memory_demand"]
    service._available = True
    service.being_provisioned = False

# Defining the inicial service placement
random_fit_services()

# Calculating user communication paths and application delays
for user in User.all():
    for application in user.applications:
        user.set_communication_path(app=application)

print("\n\n")
print("======================")
print("==== EDGE SERVERS ====")
print("======================")
for edge_server in EdgeServer.all():
    edge_server_metadata = {
        "status": edge_server.status,
        "base_station": edge_server.base_station,
        "capacity": [edge_server.cpu, edge_server.memory, edge_server.disk],
        "demand": [edge_server.cpu_demand, edge_server.memory_demand, edge_server.disk_demand],
        "services": [service.id for service in edge_server.services],
        "patch_time": edge_server.patch_time,
    }
    print(f"{edge_server}. {edge_server_metadata}")

print("\n\n")
print("======================")
print("==== APPLICATIONS ====")
print("======================")
for application in Application.all():
    user = application.users[0]
    service = application.services[0]
    image = ContainerImage.find_by(attribute_name="digest", attribute_value=service.image_digest)
    application_metadata = {
        "sla": user.delay_slas[str(application.id)],
        "delay": user.delays[str(application.id)],
        "image": image.name,
        "state": service.state,
        "demand": [service.cpu_demand, service.memory_demand],
        "host": service.server,
    }
    print(f"{application}. {application_metadata}")


##########################
#### DATASET ANALYSIS ####
##########################
# Calculating the network delay between users and edge servers (useful for defining reasonable delay SLAs)
users = []
for user in User.all():
    user_metadata = {"object": user, "all_delays": []}
    edge_servers = []
    for edge_server in EdgeServer.all():
        path = nx.shortest_path(
            G=Topology.first(), source=user.base_station.network_switch, target=edge_server.network_switch, weight="delay"
        )
        user_metadata["all_delays"].append(Topology.first().calculate_path_delay(path=path))
    user_metadata["min_delay"] = min(user_metadata["all_delays"])
    user_metadata["max_delay"] = max(user_metadata["all_delays"])
    user_metadata["avg_delay"] = sum(user_metadata["all_delays"]) / len(user_metadata["all_delays"])
    user_metadata["delays"] = {}
    for delay in sorted(list(set(user_metadata["all_delays"]))):
        user_metadata["delays"][delay] = user_metadata["all_delays"].count(delay)

    users.append(user_metadata)

print("\n\n")
print("=================================================================")
print("==== NETWORK DISTANCE (DELAY) BETWEEN USERS AND EDGE SERVERS ====")
print("=================================================================")
for user_metadata in users:
    user_attrs = {
        "object": user_metadata["object"],
        "sla": user_metadata["object"].delay_slas[str(user_metadata["object"].applications[0].id)],
        "min": user_metadata["min_delay"],
        "max": user_metadata["max_delay"],
        "avg": round(user_metadata["avg_delay"]),
        "delays": user_metadata["delays"],
    }
    print(f"{user_attrs}")
    if user_attrs["min"] > user_attrs["sla"]:
        print(f"\n\n\n\nWARNING: {user_attrs['object']} delay SLA is not achievable!\n\n\n\n")

# Calculating the infrastructure occupation and information about the services
edge_server_cpu_capacity = 0
edge_server_memory_capacity = 0
service_cpu_demand = 0
service_memory_demand = 0

for edge_server in EdgeServer.all():
    edge_server_cpu_capacity += edge_server.cpu
    edge_server_memory_capacity += edge_server.memory

for service in Service.all():
    service_cpu_demand += service.cpu_demand
    service_memory_demand += service.memory_demand

overall_cpu_occupation = round((service_cpu_demand / edge_server_cpu_capacity) * 100, 1)
overall_memory_occupation = round((service_memory_demand / edge_server_memory_capacity) * 100, 1)

print("\n\n")
print("============================================")
print("============================================")
print("==== INFRASTRUCTURE OCCUPATION OVERVIEW ====")
print("============================================")
print("============================================")
print(f"Edge Servers: {EdgeServer.count()}")
print(f"\tCPU Capacity: {edge_server_cpu_capacity}")
print(f"\tRAM Capacity: {edge_server_memory_capacity}")
print(f"Services: {Service.count()}")
print(f"\tCPU Demand: {service_cpu_demand}")

print(f"\nOverall Occupation")
print(f"\tCPU: {overall_cpu_occupation}%")
print(f"\tRAM: {overall_memory_occupation}%")

###########################
#### DATASET EXPORTING ####
###########################
# Overriding the object exporting structure to allow EdgeSimPy to import maintenance-related attributes
EdgeServer._to_dict = edge_server_to_dict
EdgeServer.collect = edge_server_collect
EdgeServer.update = edge_server_update
EdgeServer.step = edge_server_step

# Exporting scenario
ComponentManager.export_scenario(save_to_file=True, file_name="dataset1")
display_topology(Topology.first())
