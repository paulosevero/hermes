""" Contains edge-server-related functionality."""
# EdgeSimPy components
from edge_sim_py.components.network_flow import NetworkFlow
from edge_sim_py.components.container_registry import ContainerRegistry

# Python libraries
import networkx as nx


def edge_server_to_dict(self) -> dict:
    """Method that overrides the way the object is formatted to JSON."

    Returns:
        dict: JSON-friendly representation of the object as a dictionary.
    """
    dictionary = {
        "attributes": {
            "id": self.id,
            "available": self.available,
            "model_name": self.model_name,
            "cpu": self.cpu,
            "memory": self.memory,
            "disk": self.disk,
            "cpu_demand": self.cpu_demand,
            "memory_demand": self.memory_demand,
            "disk_demand": self.disk_demand,
            "coordinates": self.coordinates,
            "max_concurrent_layer_downloads": self.max_concurrent_layer_downloads,
            "active": self.active,
            "patch_time": self.patch_time,
            "status": self.status,
            "patching_log": self.patching_log,
        },
        "relationships": {
            "base_station": {"class": type(self.base_station).__name__, "id": self.base_station.id} if self.base_station else None,
            "network_switch": {"class": type(self.network_switch).__name__, "id": self.network_switch.id}
            if self.network_switch
            else None,
            "services": [{"class": type(service).__name__, "id": service.id} for service in self.services],
            "container_layers": [{"class": type(layer).__name__, "id": layer.id} for layer in self.container_layers],
            "container_images": [{"class": type(image).__name__, "id": image.id} for image in self.container_images],
            "container_registries": [{"class": type(reg).__name__, "id": reg.id} for reg in self.container_registries],
        },
    }
    return dictionary


def edge_server_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """
    metrics = {
        "Instance ID": self.id,
        "Coordinates": self.coordinates,
        "Available": self.available,
        "CPU": self.cpu,
        "RAM": self.memory,
        "Disk": self.disk,
        "CPU Demand": self.cpu_demand,
        "RAM Demand": self.memory_demand,
        "Disk Demand": self.disk_demand,
        "Ongoing Migrations": self.ongoing_migrations,
        "Services": [service.id for service in self.services],
        "Registries": [registry.id for registry in self.container_registries],
        "Layers": [layer.instruction for layer in self.container_layers],
        "Images": [image.name for image in self.container_images],
        "Download Queue": [f.metadata["object"].instruction for f in self.download_queue],
        "Waiting Queue": [layer.instruction for layer in self.waiting_queue],
        "Max. Concurrent Layer Downloads": self.max_concurrent_layer_downloads,
        "Patch Time": self.patch_time,
        "Status": self.status,
        "Patching Log": self.patching_log,
    }
    return metrics


def edge_server_update(self):
    """Method that starts the edge server's patching procedure."""
    # Setting the server status as unavailable as it cannot host applications during patching
    self.available = False

    # Recording the simulation time step where the patching procedure started
    self.patching_log = {
        "started_at": self.model.schedule.steps + 1,
        "finished_at": None,
    }

    # Updating the server status
    self.status = "being_updated"


def edge_server_step(self):
    """Method that executes the events involving the object at each time step."""
    # Setting the server as available and updated when its patching procedure is done
    if self.status == "being_updated" and self.patching_log["started_at"] + self.patch_time - 1 == self.model.schedule.steps:
        self.available = True
        self.patching_log["finished_at"] = self.model.schedule.steps + 1
        self.status = "updated"

    while len(self.waiting_queue) > 0 and len(self.download_queue) < self.max_concurrent_layer_downloads:
        layer = self.waiting_queue.pop(0)

        # Gathering the list of registries that have the layer
        registries_with_layer = []
        for registry in [reg for reg in ContainerRegistry.all() if reg.available]:
            # Checking if the registry is hosted on a valid host in the infrastructure and if it has the layer we need to pull
            if registry.server and any(layer.digest == l.digest for l in registry.server.container_layers):
                # Selecting a network path to be used to pull the layer from the registry
                path = nx.shortest_path(
                    G=self.model.topology,
                    source=registry.server.base_station.network_switch,
                    target=self.base_station.network_switch,
                )

                registries_with_layer.append({"object": registry, "path": path})

        # Selecting the registry from which the layer will be pulled to the (target) edge server
        registries_with_layer = sorted(registries_with_layer, key=lambda r: len(r["path"]))
        registry = registries_with_layer[0]["object"]
        path = registries_with_layer[0]["path"]

        # Creating the flow object
        flow = NetworkFlow(
            topology=self.model.topology,
            source=registry.server,
            target=self,
            start=self.model.schedule.steps + 1,
            path=path,
            data_to_transfer=layer.size,
            metadata={"type": "layer", "object": layer, "container_registry": registry},
        )
        self.model.initialize_agent(agent=flow)

        # Adding the created flow to the edge server's download queue
        self.download_queue.append(flow)
