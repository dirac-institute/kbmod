import json
from sys import argv
from kbmodpy import kbmod as kb

def setup_search(path, tfile):
    with open(path, 'r') as fp:
        params = json.load(fp)
    print("Loading trajectories from " + tfile + "... ", end="")
    t_list = kb.load_trajectories(tfile)#params['trajectories_file'])
    print("Done.")
    psf = kb.psf(params['psf_sigma'])
    print("Generating images... ", end="", flush=True)
    imgs = []
    for i in range(params['img_count']):
        time = i/(params['img_count']-1)
        im = kb.layered_image('img'+str(i), 
            params['x_dim'],
            params['y_dim'],
            params['noise'],
            params['noise']*params['noise'],
            time)
        for t in t_list:
            im.add_object(t.x+time*t.x_v, t.y+time*t.y_v, t.flux, psf)
        imgs.append(im)

    stack = kb.image_stack(imgs)
    del imgs
    search = kb.stack_search(stack, psf)
    search.set_debug(True)
    del stack
    print("Done.")
    return search, t_list, params

def grid_search(path, tfile):
    search, t_list, params = setup_search(path, tfile)
    search.gpu( 
        params['angle_steps'],
        params['velocity_steps'],
        params['min_angle'],
        params['max_angle'],
        params['min_vel'],
        params['max_vel'],
        int(params['img_count']/2))
    return search.get_results(0, 10000), t_list

def region_search(path, tfile):
    search, t_list, params = setup_search(path, tfile)
    results = search.region_search(
        params['x_vel'],
        params['y_vel'],
        params['radius'],
        params['min_lh'],
        int(params['img_count']/2))
    return kb.region_to_grid(results, 1.0), t_list

def run(path, tfile, search_type):
    with open(path, 'r') as fp:
        params = json.load(fp)
    if (search_type == 'grid'):
        results, t_list = grid_search(path, tfile)
    elif (search_type == 'region'):
        results, t_list = region_search(path, tfile)
    else:
        raise ValueError("Search type must be either 'grid' or 'region'")

    matched, unmatched = kb.match_trajectories(
        results, 
        t_list, 
        params['vel_error'],
        params['pixel_error']
        )
    print("\nFound " + str(len(matched)) + "/" + str(len(t_list)) + " Objects")
    print("Computed Likelihoods: " + str([t.lh for t in matched]))
    print("Missed: " + str(unmatched))
    print("Score: " + str(kb.score_results(results, t_list, 
        params['vel_error'], params['pixel_error'])))
    # print timings

if __name__ == '__main__':
    script, pfile, tfile, search_type = argv
    run(pfile, tfile, search_type)
