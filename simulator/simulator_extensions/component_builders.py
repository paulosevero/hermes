# EdgeSimPy components
from edge_sim_py.components import *

PRINT_METADATA = False


def immobile(user: object):
    user.coordinates_trace.extend([user.base_station.coordinates for _ in range(5000)])


def create_relationship_callable(relationship_value):
    return globals()[relationship_value]


def create_relationship_list_of_component_instances(relationship_value):
    attribute_values = [globals()[item["class"]].find_by_id(item["id"]) for item in relationship_value]
    return attribute_values


def create_relationship_single_component_instance(relationship_value):
    return globals()[relationship_value["class"]].find_by_id(relationship_value["id"])


def create_relationship_dictionary_of_components(relationship_value):
    attribute = {}
    for k, v in relationship_value.items():
        obj = globals()[v["class"]].find_by_id(v["id"])
        attribute[k] = obj
    return attribute


class NetworkSwitchBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.coordinates = component_instance.attributes["coordinates"]
        component_instance.active = component_instance.attributes["active"]
        component_instance.power_model_parameters = component_instance.attributes["power_model_parameters"]

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "id": component_instance.id,
                "coordinates": component_instance.coordinates,
                "active": component_instance.active,
                "power_model_parameters": component_instance.power_model_parameters,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.power_model = create_relationship_callable(component_instance.relationships["power_model"])
        component_instance.edge_servers = create_relationship_list_of_component_instances(
            component_instance.relationships["edge_servers"]
        )
        component_instance.links = create_relationship_list_of_component_instances(component_instance.relationships["links"])
        component_instance.base_station = create_relationship_single_component_instance(
            component_instance.relationships["base_station"]
        )

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "power_model": component_instance.power_model,
                "edge_servers": component_instance.edge_servers,
                "links": component_instance.links,
                "base_station": component_instance.base_station,
            }
            print(metadata)


class NetworkLinkBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.delay = component_instance.attributes["delay"]
        component_instance.bandwidth = component_instance.attributes["bandwidth"]
        component_instance.bandwidth_demand = component_instance.attributes["bandwidth_demand"]
        component_instance.active = component_instance.attributes["active"]

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "id": component_instance.id,
                "delay": component_instance.delay,
                "bandwidth": component_instance.bandwidth,
                "bandwidth_demand": component_instance.bandwidth_demand,
                "active": component_instance.active,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.topology = create_relationship_single_component_instance(component_instance.relationships["topology"])
        component_instance.active_flows = create_relationship_list_of_component_instances(
            component_instance.relationships["active_flows"]
        )
        component_instance.nodes = create_relationship_list_of_component_instances(component_instance.relationships["nodes"])

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "topology": component_instance.topology,
                "active_flows": component_instance.active_flows,
                "nodes": component_instance.nodes,
            }
            print(metadata)


class BaseStationBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.coordinates = component_instance.attributes["coordinates"]
        component_instance.wireless_delay = component_instance.attributes["wireless_delay"]

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "id": component_instance.id,
                "coordinates": component_instance.coordinates,
                "wireless_delay": component_instance.wireless_delay,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.users = create_relationship_list_of_component_instances(component_instance.relationships["users"])
        component_instance.edge_servers = create_relationship_list_of_component_instances(
            component_instance.relationships["edge_servers"]
        )
        component_instance.network_switch = create_relationship_single_component_instance(
            component_instance.relationships["network_switch"]
        )

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "users": component_instance.users,
                "edge_servers": component_instance.edge_servers,
                "network_switch": component_instance.network_switch,
            }
            print(metadata)


class UserBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.coordinates = component_instance.attributes["coordinates"]
        component_instance.coordinates_trace = component_instance.attributes["coordinates_trace"]
        component_instance.delays = component_instance.attributes["delays"]
        component_instance.delay_slas = component_instance.attributes["delay_slas"]
        component_instance.communication_paths = component_instance.attributes["communication_paths"]
        component_instance.making_requests = component_instance.attributes["making_requests"]
        component_instance.mobility_model_parameters = component_instance.attributes["mobility_model_parameters"]

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "id": component_instance.id,
                "coordinates": component_instance.coordinates,
                "coordinates_trace": component_instance.coordinates_trace,
                "delays": component_instance.delays,
                "delay_slas": component_instance.delay_slas,
                "communication_paths": component_instance.communication_paths,
                "making_requests": component_instance.making_requests,
                "mobility_model_parameters": component_instance.mobility_model_parameters,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.access_patterns = create_relationship_dictionary_of_components(
            component_instance.relationships["access_patterns"]
        )
        component_instance.mobility_model = create_relationship_callable(component_instance.relationships["mobility_model"])
        component_instance.applications = create_relationship_list_of_component_instances(
            component_instance.relationships["applications"]
        )
        component_instance.base_station = create_relationship_single_component_instance(
            component_instance.relationships["base_station"]
        )

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "access_patterns": component_instance.access_patterns,
                "mobility_model": component_instance.mobility_model,
                "applications": component_instance.applications,
                "base_station": component_instance.base_station,
            }
            print(metadata)


class ContainerLayerBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.digest = component_instance.attributes["digest"]
        component_instance.size = component_instance.attributes["size"]
        component_instance.instruction = component_instance.attributes["instruction"]

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "id": component_instance.id,
                "digest": component_instance.digest,
                "size": component_instance.size,
                "instruction": component_instance.instruction,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.server = create_relationship_single_component_instance(component_instance.relationships["server"])

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "server": component_instance.server,
            }
            print(metadata)


class ContainerImageBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.name = component_instance.attributes["name"]
        component_instance.tag = component_instance.attributes["tag"]
        component_instance.digest = component_instance.attributes["digest"]
        component_instance.layers_digests = component_instance.attributes["layers_digests"]

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "id": component_instance.id,
                "name": component_instance.name,
                "tag": component_instance.tag,
                "digest": component_instance.digest,
                "layers_digests": component_instance.layers_digests,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.server = create_relationship_single_component_instance(component_instance.relationships["server"])

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "server": component_instance.server,
            }
            print(metadata)


class ServiceBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.label = component_instance.attributes["label"]
        component_instance.state = component_instance.attributes["state"]
        component_instance._available = component_instance.attributes["_available"]
        component_instance.cpu_demand = component_instance.attributes["cpu_demand"]
        component_instance.memory_demand = component_instance.attributes["memory_demand"]
        component_instance.image_digest = component_instance.attributes["image_digest"]

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "id": component_instance.id,
                "label": component_instance.label,
                "state": component_instance.state,
                "_available": component_instance._available,
                "cpu_demand": component_instance.cpu_demand,
                "memory_demand": component_instance.memory_demand,
                "image_digest": component_instance.image_digest,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.application = create_relationship_single_component_instance(component_instance.relationships["application"])
        component_instance.server = create_relationship_single_component_instance(component_instance.relationships["server"])

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "application": component_instance.application,
                "server": component_instance.server,
            }
            print(metadata)


class ContainerRegistryBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.cpu_demand = component_instance.attributes["cpu_demand"]
        component_instance.memory_demand = component_instance.attributes["memory_demand"]

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "id": component_instance.id,
                "cpu_demand": component_instance.cpu_demand,
                "memory_demand": component_instance.memory_demand,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.server = create_relationship_single_component_instance(component_instance.relationships["server"])

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "server": component_instance.server,
            }
            print(metadata)


class ApplicationBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.label = component_instance.attributes["label"]

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "id": component_instance.id,
                "label": component_instance.label,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.services = create_relationship_list_of_component_instances(component_instance.relationships["services"])
        component_instance.users = create_relationship_list_of_component_instances(component_instance.relationships["users"])

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "services": component_instance.services,
                "users": component_instance.users,
            }
            print(metadata)


class EdgeServerBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.available = component_instance.attributes["available"]
        component_instance.model_name = component_instance.attributes["model_name"]
        component_instance.cpu = component_instance.attributes["cpu"]
        component_instance.memory = component_instance.attributes["memory"]
        component_instance.disk = component_instance.attributes["disk"]
        component_instance.cpu_demand = component_instance.attributes["cpu_demand"]
        component_instance.memory_demand = component_instance.attributes["memory_demand"]
        component_instance.disk_demand = component_instance.attributes["disk_demand"]
        component_instance.coordinates = component_instance.attributes["coordinates"]
        component_instance.max_concurrent_layer_downloads = component_instance.attributes["max_concurrent_layer_downloads"]
        component_instance.active = component_instance.attributes["active"]
        component_instance.patch_time = component_instance.attributes["patch_time"]
        component_instance.status = component_instance.attributes["status"]
        component_instance.patching_log = component_instance.attributes["patching_log"]

        if PRINT_METADATA:
            metadata = {
                "id": component_instance.id,
                "available": component_instance.available,
                "model_name": component_instance.model_name,
                "cpu": component_instance.cpu,
                "memory": component_instance.memory,
                "disk": component_instance.disk,
                "cpu_demand": component_instance.cpu_demand,
                "memory_demand": component_instance.memory_demand,
                "disk_demand": component_instance.disk_demand,
                "coordinates": component_instance.coordinates,
                "max_concurrent_layer_downloads": component_instance.max_concurrent_layer_downloads,
                "active": component_instance.active,
                "patch_time": component_instance.patch_time,
                "status": component_instance.status,
                "patching_log": component_instance.patching_log,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.base_station = create_relationship_single_component_instance(
            component_instance.relationships["base_station"]
        )
        component_instance.network_switch = create_relationship_single_component_instance(
            component_instance.relationships["network_switch"]
        )
        component_instance.services = create_relationship_list_of_component_instances(component_instance.relationships["services"])
        component_instance.container_layers = create_relationship_list_of_component_instances(
            component_instance.relationships["container_layers"]
        )
        component_instance.container_images = create_relationship_list_of_component_instances(
            component_instance.relationships["container_images"]
        )
        component_instance.container_registries = create_relationship_list_of_component_instances(
            component_instance.relationships["container_registries"]
        )

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "base_station": component_instance.base_station,
                "network_switch": component_instance.network_switch,
                "services": component_instance.services,
                "container_layers": component_instance.container_layers,
                "container_images": component_instance.container_images,
                "container_registries": component_instance.container_registries,
            }
            print(metadata)


class CircularDurationAndIntervalAccessPatternBuilder:
    @classmethod
    def create_attributes(cls, component_instance, attributes_metadata):
        component_instance.id = component_instance.attributes["id"]
        component_instance.duration_values = component_instance.attributes["duration_values"]
        component_instance.interval_values = component_instance.attributes["interval_values"]
        component_instance.history = component_instance.attributes["history"]

        if PRINT_METADATA:
            metadata = {
                "object": component_instance,
                "id": component_instance.id,
                "duration_values": component_instance.duration_values,
                "interval_values": component_instance.interval_values,
                "history": component_instance.history,
            }
            print(metadata)

        return component_instance

    @classmethod
    def create_relationships(cls, component_instance):
        component_instance.user = create_relationship_single_component_instance(component_instance.relationships["user"])
        component_instance.app = create_relationship_single_component_instance(component_instance.relationships["app"])

        metadata = {
            "object": component_instance,
            "user": component_instance.user,
            "app": component_instance.app,
        }
        if PRINT_METADATA:
            print(metadata)
