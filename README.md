# Latency-Aware Maintenance Policy for Edge Computing Infrastructures

## Scenario

- Edge servers with heterogeneous capacity
- Users accessing applications with distinct delay SLAs
- Images with a consirable number of layers that conflict with the downloading limit of edge servers

## Addressed Challenges

- **5.2 Optimized Prioritization of Maintenance Decisions (vulnerability surface v2)**
  - The Vulnerability Surface is given by the product of the number of outdated servers and the elapsed maintenance time.

  It works great on homogeneous cloud data centers. However, the number of outdated servers does not necessarily represent the actual vulnerability surface of heterogeneous infrastructures.

  We could prioritize migrations that safeguard more applications early rather than trying to update servers early. In some scenarios, updating a few high-capacity servers would safeguard most applications.

- **5.5 User Location Awareness during Migration Decisions**
  - Perform concurrent migrations to reduce the overall maintenance time (THIS IS A SYSTEM MODEL APPROACH, NOT A POLICY FEATURE)

- **5.6 Optimized Provisioning of Containerized Applications**
  - Reduce the maintenance time by taking migration decisions that consider the shared portion of application container images


## Key Decisions

**Heuristic Decisions:**
=> Amount of outdated computational capacity drained
=> Number of services migrated to updated servers
=> Amount of cached data from container images of migrated services
=> Number of delay SLA violations

**NSGA-II Decisions:**
=> Maintenance Time
=> Delay SLA Violations


**1. Who to Drain:**
- Servers with larger capacity
- Servers hosting applications with:
    - Stateless services (aggregated size of service states that would have to be migrated)
    - Images with increased layer size cached from upgraded servers with available capacity to host the service

**2. Where to Migrate:**
1. Servers updated
2. Servers close enough to users (delay SLA)
3. Servers with cached layers (vulnerability surface v2)
4. Servers with higher available capacity (higher probability of being able to use the same server w/ its cached layers in the future to host other services and potentially avoid prov. time SLA violations). Pay attention to avoid conflict with (3)

**3. Migration Order:**
1. Services with tight delay SLA
2. Services with low CPU/RAM demand that use images that contain popular layers (provioning those services images first can potentially benefit other services that are scheduled to be drained, potentially to the same host)
3. **ADVANCED:** create a dataset where images have lots of layers so prioritizing the migration of services with tight prov. time SLAs is key to avoid those services being affected by waiting times due to the limit of concurrent downloads within hosts


##################################
#### LIST OF CONTAINER IMAGES ####
##################################
01. centos
02. ros
03. debian
04. ruby
05. rust
06. python
07. erlang
08. elixir
09. telegraf
10. storm
11. node
12. tomcat
13. nginx
14. redis
15. mongo

#############################
#### APPLICATION CLASSES ####
#############################
==> 1. OPERATING SYSTEMS
1.  centos
2.  ros
3.  debian

==> 2. PROGRAMMING LANGUAGES
04. ruby
05. rust
06. python
07. erlang
08. elixir

==> 3. MONITORING
09. telegraf

==> 4. STREAMING
10. storm

==> 5. WEB APPLICATIONS
11. node
12. tomcat
13. nginx

==> 6. DATABASES
14. redis
15. mongo

## Task List

- [x] Format simulator's output to ease the exporting to CSV
- [x] Create spreadsheet with dataset parameters and scaffold for results
- [x] Create script to run experiments at scale (run_experiments.py)
- [ ] Refactor the dataset creation script
- [ ] Define and create large-scale dataset
- [ ] Run experiments (sensitivity analysis and comparison against baseline)
- [ ] Collect results
- [ ] Create graphs
