""" Contains edge-server-related functionality."""
# EdgeSimPy components
from edge_sim_py.components.container_image import ContainerImage
from edge_sim_py.components.container_layer import ContainerLayer


def service_collect(self) -> dict:
    """Method that collects a set of metrics for the object.

    Returns:
        metrics (dict): Object metrics.
    """

    if len(self._Service__migrations) > 0:

        last_migration = {
            "status": self._Service__migrations[-1]["status"],
            "origin": str(self._Service__migrations[-1]["origin"]),
            "target": str(self._Service__migrations[-1]["target"]),
            "start": self._Service__migrations[-1]["start"],
            "end": self._Service__migrations[-1]["end"],
            "waiting": self._Service__migrations[-1]["waiting_time"],
            "pulling": self._Service__migrations[-1]["pulling_layers_time"],
            "migr_state": self._Service__migrations[-1]["migrating_service_state_time"],
        }
    else:
        last_migration = None

    if not hasattr(self, "time_steps_on_outdated_hosts"):
        self.time_steps_on_outdated_hosts = 0
    if not hasattr(self, "time_steps_on_updated_hosts"):
        self.time_steps_on_updated_hosts = 0

    if self.server and self.server.status == "outdated":
        self.time_steps_on_outdated_hosts += 1
    else:
        self.time_steps_on_updated_hosts += 1

    metrics = {
        "Instance ID": self.id,
        "Available": self._available,
        "Server": self.server.id if self.server else None,
        "Being Provisioned": self.being_provisioned,
        "Last Migration": last_migration,
        "Time Steps on Outdated Hosts": self.time_steps_on_outdated_hosts,
        "Time Steps on Updated Hosts": self.time_steps_on_updated_hosts,
    }
    return metrics


def service_provision(self, target_server: object):
    """Starts the service's provisioning process. This process comprises both placement and migration. In the former, the
    service is not initially hosted by any server within the infrastructure. In the latter, the service is already being
    hosted by a server and we want to relocate it to another server within the infrastructure.

    Args:
        target_server (object): Target server.
    """
    # Gathering layers present in the target server (layers, download_queue, waiting_queue)
    layers_downloaded = [layer for layer in target_server.container_layers]
    layers_on_download_queue = [flow.metadata["object"] for flow in target_server.download_queue]
    layers_on_waiting_queue = [layer for layer in target_server.waiting_queue]

    layers_on_target_server = layers_downloaded + layers_on_download_queue + layers_on_waiting_queue

    # Gathering the list of layers that compose the service image that are not present in the target server
    image = ContainerImage.find_by(attribute_name="digest", attribute_value=self.image_digest)

    sizes_of_uncached_layers = []
    sizes_of_cached_layers = []

    for layer_digest in image.layers_digests:
        # As the image only stores its layers digests, we need to get information about each of its layers
        layer_metadata = ContainerLayer.find_by(attribute_name="digest", attribute_value=layer_digest)
        if not any(layer.digest == layer_digest for layer in layers_on_target_server):

            sizes_of_uncached_layers.append(layer_metadata.size)

            # Creating a new layer object that will be pulled to the target server
            layer = ContainerLayer(
                digest=layer_metadata.digest,
                size=layer_metadata.size,
                instruction=layer_metadata.instruction,
            )
            self.model.initialize_agent(agent=layer)

            # Reserving the layer disk demand inside the target server
            target_server.disk_demand += layer.size

            # Adding the layer to the target server's waiting queue (layers it must download at some point)
            target_server.waiting_queue.append(layer)
        else:
            sizes_of_cached_layers.append(layer_metadata.size)

    # Telling EdgeSimPy that this service is being provisioned
    self.being_provisioned = True

    # Telling EdgeSimPy the service's current server is now performing a migration. This action is only triggered in case
    # this method is called for performing a migration (i.e., the service is already within the infrastructure)
    if self.server:
        self.server.ongoing_migrations += 1

    # Reserving the service demand inside the target server and telling EdgeSimPy that server will receive a service
    target_server.ongoing_migrations += 1
    target_server.cpu_demand += self.cpu_demand
    target_server.memory_demand += self.memory_demand

    # Updating the service's migration status
    self._Service__migrations.append(
        {
            "status": "waiting",
            "maintenance_batch": self.model.maintenance_batches,
            "origin": self.server,
            "target": target_server,
            "start": self.model.schedule.steps + 1,
            "end": None,
            "waiting_time": 0,
            "pulling_layers_time": 0,
            "migrating_service_state_time": 0,
            "cache_hits": len(sizes_of_cached_layers),
            "cache_misses": len(sizes_of_uncached_layers),
            "sizes_of_cached_layers": sizes_of_cached_layers,
            "sizes_of_uncached_layers": sizes_of_uncached_layers,
        }
    )
