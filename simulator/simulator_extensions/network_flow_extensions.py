""" Contains network-flow-related functionality."""


def network_flow_step(self):
    """Method that executes the events involving the object at each time step."""
    if self.status == "active":
        # Updating the flow progress according to the available bandwidth
        if not any([bw == None for bw in self.bandwidth.values()]):
            self.data_to_transfer -= min(self.bandwidth.values())

        if self.data_to_transfer <= 0:
            # Updating the completed flow's properties
            self.data_to_transfer = 0

            # Storing the current step as when the flow ended
            self.end = self.model.schedule.steps + 1

            # Updating the flow status to "finished"
            self.status = "finished"

            # Releasing links used by the completed flow
            for i in range(0, len(self.path) - 1):
                link = self.model.topology[self.path[i]][self.path[i + 1]]
                link["active_flows"].remove(self)

            # When container layer flows finish: Adds the container layer to its target host
            if self.metadata["type"] == "layer":
                # Removing the flow from its target host's download queue
                self.target.download_queue.remove(self)

                # Adding the layer to its target host
                layer = self.metadata["object"]
                layer.server = self.target
                self.target.container_layers.append(layer)

                # Decreasing the number of active flows within the container registry used to pull the layer
                self.metadata["container_registry"].active_flows -= 1

            # When service state flows finish: change the service migration status
            elif self.metadata["type"] == "service_state":
                service = self.metadata["object"]
                service._Service__migrations[-1]["status"] = "finished"
