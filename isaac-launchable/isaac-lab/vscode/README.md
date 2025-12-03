# Isaac Launchable Usage

This project provides a simple way to interact with Isaac Lab, and Isaac Sim, either on the cloud or locally.
See the [project repo](https://github.com/isaac-sim/isaac-launchable) for more detaild instructions, but below is a quickstart guide.

## Running Isaac Lab

You can run any of the Issac Lab scripts with the streaming Isaac Sim experience with the following command:

```console
./isaaclab/isaaclab.sh -p "isaaclab/scripts/tutorials/00_sim/create_empty.py" --kit_args="--no-window --enable omni.kit.livestream.webrtc"
```

Then, in a separate browser window, open `http://{your instance's address}/viewer`.

To run any other Isaac Lab commands, simply append those same arguments as shown above: `--kit_args="--no-window --enable omni.kit.livestream.webrtc"`

## Running Isaac Sim

You can run the streaming Isaac Sim application at anytime with the following command:

```console
./isaaclab/_isaac_sim/isaac-sim.sh --no-window --enable omni.kit.livestream.webrtc
```

Then, in a separate browser window, open `https://{your instance's address}/viewer`.

For example, if this is run on a local computer: `127.0.0.1/viewer`
If this is run on an AWS instance, the address may be more like: 
`http://ec2-00-00-000-000.compute-1.amazonaws.com/viewer`
