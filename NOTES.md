**Heuristic Decisions:**
=> Amount of outdated computational capacity drained
=> Number of services migrated to updated servers
=> Amount of cached data from container images of migrated services
=> Number of delay SLA violations

**NSGA-II Decisions:**
=> Maintenance Time
=> Delay SLA Violations

**METRICS OF INTEREST**
=> Maintenance time
=> Provisioning time (depicting time values for: waiting, pulling, state migration)
=> CDF of server updated capacity throughout the simulation
=> CDF of services hosted by updated servers
=> Number of delay SLA violations

**OPTIMAL:** [1, 1, 8, 8, 8, 8, 7, 7, 3, 3, 3, 3, 7, 7, 3, 3, 3, 3]
    BATCH 1 => [1, 1, 8, 8, 8, 8]
        => Updating: ES2, ES3, ES5, ES6, ES7
        => Migrating:

    BATCH 2 => [7, 7, 3, 3, 3, 3]
        => Updating: 
        => Migrating: S1, S2, S3, S4, S5, S6

    BATCH 3 => [7, 7, 3, 3, 3, 3]
        => Updating: ES1, ES8
        => Migrating: 
