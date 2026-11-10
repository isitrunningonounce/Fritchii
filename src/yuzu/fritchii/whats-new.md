<!--
Fritchii's "What's new" notes for this build. The app shows them after an update, and
tools/release-local.ps1 publishes them as the GitHub release notes under the download links, so
write them before the release build. Keep a "---" line out of the first 20 lines: the app drops
whatever is above one when it mentions downloads. While this file holds only this comment, no
"What's new" window pops up after an update (Help > What's New says there are no notes yet).
-->
This build is Fritchii v5.6 with this fork's fixes for Yoshi and the Mysterious Book, Mario Kart World and Star Fox. Expect one full "Preparing shaders" run after updating (see v5.6 below).

### Fixed in this build
- **Yoshi and the Mysterious Book renders at full resolution.** The game draws gameplay at a fixed 50% (960x540 docked) and drops its menus to dynamic resolution (about 1400x788); both now render at your full output resolution. It's built in and on by default, so the "Full Resolution" mod isn't needed any more. Turn "Max Resolution (built-in)" off in the game's Add-Ons to go back.
- **Yoshi and the Mysterious Book: gameplay shaders the emulator couldn't translate now run**: the colour lookup pass, texture gathers through bindless handles and several compute shaders (ported from Eglantine #762 and upstream nx2).
- **Mario Kart World: characters are no longer dark.** The lighting probes the game reads were cut off by an 8 MiB limit on GPU storage buffers.
- **Star Fox runs the game's own upscaler** instead of a stand-in: the images it writes now reach the buffers it reads them back from. **Its opening cinematic is back to 30 fps** (it had dropped to 19).
- **Videos keep the right colours** after one switches from limited back to full colour range. Its frames were still converted as limited, which crushed blacks and blew out whites.

### New in this build
- **Updates come from this fork:** builds are named build-N and check this repository's releases for the next one.
- **Linux:** an AppImage that uses Wayland when it's available (X11 otherwise), and a tar.gz that uses your system's libraries.

### Fritchii v5.6

**Heads up: "Preparing shaders" runs once more** (around 15 minutes on an 8-core CPU), on every GPU, because of the AMD horizon fix below and a new detail Fritchii now stores about each shader. After that it's straight into the game again. The first minute or two after it can also be a bit slower while the driver settles.

#### Fixed
- **The huge glowing triangles ("light beams") are much rarer.** In Kokiri Forest, Kakariko, the graveyard and Hyrule Field they used to grow over a session. One cause is fixed: a compute shader wrote further than the emulator tracked, and the game read old data back from there. Every compute shader is now checked once, and tracked fully if it writes further. No fps cost. They can still show up in long sessions: if you see one, use Help > Export Log and post it in the Issues channel.
- **AMD: no more GPU hangs on draws the game makes with mesh shaders** (zortos293/Lichii#814). AMD's driver now goes through a compute fallback for them. New **Mesh Shaders** setting in Graphics (Automatic, Native, Compute) if you want to try the other path.
- **AMD: no more rainbow "disco" outline** where mountains and far scenery meet the sky. Draws that don't output every colour channel left the rest undefined, and AMD's driver filled it with garbage. Port of jpkto's Lichii #353 (Fritchii #29 by trushna88).
- **The opening cutscenes and the main menu video stay in sync with their audio.** The game only shows about 30 frames a second on those screens, so the 60 fps videos played at half speed while the music kept going (almost 7 seconds behind after 12 seconds). Videos now show the frame that matches the audio and skip the rest. A broken video frame or a failing decoder can no longer crash the game.
- **The Song of Time no longer ends with a loud crackle** (zortos293/Lichii#1073 by justin-wq1337).
- **No more freezes of up to 3 seconds** when the game swaps a compute shader mid-play (zortos293/Lichii#969 by El-Sashok).
- **Unlocking the speed limit no longer makes OoT run fast.** Above 60 fps the whole game sped up; it now stays at its normal speed. Turbo and Slow still work as fast-forward and slow motion. In OoT, Ctrl+U now switches the high refresh rate mode instead (see New).
- **Opening the ocarina without a microphone no longer crashes the game** (zortos293/Lichii#497 and #881, via Fritchii #16).
- Mods that patch the same spot as Fritchii's built-in patches win over them (Fritchii #5). Game titles from NUP files show properly (Fritchii #20). Empty controller mapping values are no longer lost (Fritchii #30 by FFcustoms). Fritchii builds on Linux again (Fritchii #19).
- **Help > Export Log always sends this session's log.** A Fritchii that never finished closing (for example after an update) could keep the log file locked, and the export then sent an old log.

#### New
- **High refresh rate (experimental): press Ctrl+U in OoT to go above 60 fps** at normal game speed, and again to go back, no restart needed. The cap is in Graphics > Advanced (your monitor's refresh rate, 90, 120, 144 or 165), and the Graphics checkbox follows Ctrl+U. Water ripples keep moving above 60 too (zortos293/Lichii#1152, after #655 by atk3001). It's new: tell us if a timer, cutscene or minigame acts weird with it on. The 30 FPS mode still wins over it.
- **Dark's standalone Dynamic FPS mod is skipped and hidden automatically**, since Fritchii has it built in (the mod also blocked the high refresh mode). Your files aren't touched. If you turned the built-in off in Add-Ons to use the mod, nothing changes for you.
- **Reset to Defaults** (Settings > General) puts every setting back to Fritchii's own tuned settings, after saving a backup. **Restore Settings** brings any saved set back. Controls and game folders stay.
- **"My settings don't apply" explained:** when OoT has its own values in Configure Game, the Settings window now says so at the top, with one button to put OoT back on your global settings. Export Log lists them too.
- **RAM and VRAM in the performance overlay**, with how much your PC has left (amber when it's getting low).
- **"Shaders build again in game" reports explain themselves:** the log now shows how big the GPU driver's own shader cache is at boot, after preparing and at exit, whether the last session closed cleanly, and how fast preparing ran.
- **OoT's videos look like they should again on NVIDIA RTX.** The title screen and cutscene videos go through the game's own upscaler again (it removes banding and sharpens), instead of the plain stretch every build has used since v5. It costs nothing visible on RTX cards. GPUs that would have to emulate its math keep the fast stretch. Graphics > "Ocarina of Time movies" lets you pick either.
- **Discord and Donate buttons** on the home screen and in the Help menu.

#### Faster and lighter
- **About 1 GB less RAM:** memory the game cleared to zero is given back to Windows.
- **PCs with less than 24 GB of RAM** keep fewer old GPU pipelines around, so long sessions page less.
- **About 5 fps more in Castle Town** (45 to 50 on an RTX 4080 standing in the market): the emulator stopped sending the GPU thousands of empty submissions a second, and reports and fences no longer allocate memory each time (part of zortos293/Lichii#1161).
- **No more VRAM pile-up:** old render targets the game replaced are cleaned out, so VRAM stays around 5.5 GB in Castle Town instead of filling to 10-11 GB, and the freezes when it got full are gone (fights especially).
- **Long sessions:** pipelines rebuilt after being cleaned out keep their buffers, so a long session can't slowly break some meshes (zortos293/Lichii#1143).
- **Fewer dips in busy spots:** a depth buffer that the game clears right away is no longer copied in from game memory first (7.5 MB every frame in Castle Town, the cause of an NVIDIA player's dips). About +2.5 fps in Castle Town.
- **Texture memory on 10 GB and bigger cards** can fill further before the cleanup starts (zortos293/Lichii#1129 by Liomar Oliveira).
- **FXAA is off by default.** OoT's own anti-aliasing (or DLSS) already smooths edges, FXAA on top only blurred the picture. Fritchii's tuned settings are applied once more with this update; you can change anything afterwards.
