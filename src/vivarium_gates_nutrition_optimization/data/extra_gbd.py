from vivarium_gbd_access import (
    constants as gbd_constants,
    gbd,
)


@gbd.memory.cache
def get_gbd_hierarchy():
    from db_queries import get_location_metadata
    # location_set_id 35 is for GBD model results
    hierarchy = get_location_metadata(
        location_set_id=35,
        decomp_step=gbd_constants.DECOMP_STEP.STEP_4,
        gbd_round_id=gbd_constants.ROUND_IDS.GBD_2019,
    )
    return hierarchy

