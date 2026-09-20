/* The Minecraft demo's bot: everything that has to speak the game protocol.
 *
 * Node owns the body -- connection, world readout, pathfinding, digging,
 * crafting -- and owns no decisions.  Each tick it reads the world off
 * mineflayer's own API, enumerates the actions that are legal right now,
 * POSTs both to the Python server (/minecraft/api/decide) and executes the
 * label that comes back.
 *
 * The world is generated from a fresh random seed every run and nothing in
 * here knows anything about it.  There are no coordinates, no waypoints, no
 * stages and no progression: an action is offered when the game permits it
 * and at no other time.  The bot will flounder, starve and fall into holes.
 * That is the measurement, not a defect, so there are no rescues here.
 *
 * The server this connects to is Minecraft's code, not ours, so it is not in
 * the repo.  To reproduce it, outside the repo:
 *
 *   mkdir -p ~/mc && cd ~/mc
 *   curl -sL -o server.jar https://piston-data.mojang.com/v1/objects/\
 *   4707d00eb834b446575d89a61a11b5d548d8c001/server.jar
 *   echo 'eula=true' > eula.txt
 *   SEED=$(python3 -c 'import random; print(random.getrandbits(63))')
 *   cat > server.properties <<EOF
 *   online-mode=false
 *   gamemode=survival
 *   difficulty=easy
 *   level-seed=$SEED
 *   level-name=world-$SEED
 *   server-port=25565
 *   view-distance=10
 *   simulation-distance=8
 *   spawn-protection=0
 *   EOF
 *   java -Xmx2G -jar server.jar nogui
 *
 * That jar is Minecraft 1.21.4.  mineflayer speaks newer protocols, but
 * prismarine-viewer 1.33 ships block models and textures only up to 1.21.4
 * (see its viewer/lib/version.js), so 1.21.4 is the newest version both
 * halves of this demo support.  keepInventory is left at its default (off):
 * the bot loses its things when it dies, and that is part of the demo.
 *
 * Then, in this directory, passing the same seed so the run is reproducible:
 *
 *   npm install
 *   node bot.js --seed $SEED --mc-port 25565 --viewer-port 3000 \
 *     --python http://127.0.0.1:8000
 *
 * --seed is recorded and displayed, never read: replaying a run means
 * generating the world from it again, not telling the bot anything.
 */
'use strict';

const http = require('http');
const https = require('https');
const { URL } = require('url');
const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const mineflayerViewer = require('prismarine-viewer').mineflayer;
const Vec3 = require('vec3').Vec3;

const MINECRAFT_VERSION = '1.21.4';

/* How far the block scan reaches, horizontally and vertically. */
const SCAN_RADIUS = 32;
const SCAN_HEIGHT = 16;

/* Where the per-direction surface samples are taken, in metres along the
 * axis, each sampled at three lateral offsets. */
const PROBE_DISTANCES = [8, 16, 24, 32, 40, 48, 56, 64];
const PROBE_SPREAD = [-4, 0, 4];

/* The pathfinder's working range, not a preference: a block further than
 * this is not something the bot can walk to and break this decision. */
const GATHER_RANGE = 16;
/* Likewise for entities, which move while the bot approaches. */
const MOB_RANGE = 12;

const EXPLORE_DISTANCE = 32;
const COMPASS = {
  north: new Vec3(0, 0, -1),
  east: new Vec3(1, 0, 0),
  south: new Vec3(0, 0, 1),
  west: new Vec3(-1, 0, 0),
};

const AIR = ['air', 'cave_air', 'void_air'];

/* The bot's eyes, in metres above its feet.  bot.entity.position is at the
 * feet, and a ray cast from there is stopped by the block underfoot. */
const EYE_HEIGHT = 1.62;

/* The most entities reported at once.  A mechanical bound on the size of
 * the readout, filled outward from the bot: nearest first and nothing else
 * decides what is in it. */
const VISIBLE_LIMIT = 10;

/* How far the viewer draws, in chunks.  The server only ever sends the bot
 * view-distance chunks, so raising this past the server's own setting draws
 * nothing more. */
const VIEWER_VIEW_DISTANCE = 10;

/* Shortest gap between decisions, so an instant action cannot spin the
 * loop. */
const TICK_MS = 700;
/* How long one action gets before it is abandoned and reported as timed
 * out. */
const ACTION_TIMEOUT_MS = 20000;

/* Collecting what a break drops, inside the action's own budget: how far
 * from the broken block a drop counts as that break's, how long the drop
 * has to appear, how long the walk onto it and the pickup get, and the gap
 * between polls.  The game holds a fresh drop for ten ticks before anyone
 * may pick it up. */
const DROP_RADIUS = 6;
const DROP_SPAWN_MS = 1500;
const COLLECT_MS = 5000;
const POLL_MS = 100;

/* How long the bot is given to walk into the square it has just opened. */
const STEP_MS = 1500;

/* A pillar: how long the bot has to leave the ground, and how long it then
 * has to come to rest on whatever is underfoot. */
const JUMP_MS = 1000;
const SETTLE_MS = 1500;

/* How long one smelt waits for the furnace to produce, inside the action's
 * own budget.  The game takes ten seconds per item. */
const SMELT_WAIT_MS = 15000;

/* How long to wait, after a spawn, for the chunk the bot is standing in.
 * The server sends the spawn packet before the chunks around it, and on a
 * rejoin into existing playerdata the gap is wide enough that the world
 * readout would run against an empty world. */
const WORLD_WAIT_MS = 60000;

/* Thrown for our own faults, so the action runner lets them through instead
 * of filing them as a game outcome. */
class Bug extends Error {}

function must(condition, message) {
  if (!condition) throw new Bug(message);
}

/* Which foods the game lets you eat on a full hunger bar.  Java's own data
 * in minecraft-data does not carry the flag, but Bedrock's item components
 * do, under the same item names and with the same values, so the set is read
 * off the game's data rather than written down here. */
function alwaysEdibleNames() {
  const data = require('minecraft-data');
  const newest = data.versions.bedrock[0].minecraftVersion;
  const bedrock = data(`bedrock_${newest}`);
  const out = new Set();
  let foods = 0;
  for (const item of Object.values(bedrock.itemsByName)) {
    const components = item.nbt?.value?.components?.value;
    const food = components?.['minecraft:food']?.value;
    if (food === undefined) continue;
    foods += 1;
    if (food.can_always_eat?.value === 1) out.add(item.name);
  }
  must(foods > 0 && out.size > 0,
    `no can_always_eat data in the Bedrock items: ${foods} foods`);
  return out;
}

const ALWAYS_EDIBLE = alwaysEdibleNames();

/* What burns in a furnace.  No edition's data carries the burn times, so the
 * game's rule is written out: the named items, plus everything made of the
 * overworld woods, which the game burns by its wood tags.  Crimson and warped
 * are fungus, not wood, and the game will not burn them.
 */
const FUEL_NAMES = new Set(['coal', 'charcoal', 'coal_block', 'blaze_rod',
  'lava_bucket', 'dried_kelp_block', 'stick', 'bamboo', 'bamboo_block',
  'scaffolding', 'bowl', 'ladder', 'bookshelf', 'chest', 'crafting_table',
  'barrel', 'note_block', 'jukebox', 'daylight_detector', 'composter',
  'cartography_table', 'fletching_table', 'lectern', 'loom',
  'smithing_table', 'dead_bush', 'azalea', 'flowering_azalea']);

const WOOD_TYPES = ['oak', 'spruce', 'birch', 'jungle', 'acacia', 'dark_oak',
  'mangrove', 'cherry', 'pale_oak', 'bamboo'];

const WOODEN_SUFFIXES = ['_planks', '_log', '_wood', '_sapling', '_slab',
  '_stairs', '_fence', '_fence_gate', '_door', '_trapdoor', '_button',
  '_pressure_plate', '_sign', '_hanging_sign', '_boat', '_chest_boat'];

function isWooden(name) {
  const wood = WOOD_TYPES.some((type) => name.startsWith(`${type}_`)
    || name.startsWith(`stripped_${type}_`));
  return wood && WOODEN_SUFFIXES.some((suffix) => name.endsWith(suffix));
}

function isFuel(name) {
  if (FUEL_NAMES.has(name)) return true;
  if (name.endsWith('_wool') || name.endsWith('_carpet')) return true;
  return isWooden(name);
}

/* What a furnace turns into what.  Java's minecraft-data carries crafting
 * recipes only; Bedrock's carry the furnace ones, under the same item names
 * and with the same inputs and outputs -- except where Bedrock still keeps
 * the pre-flattening aggregate names (log, log2, wood), which no Java item
 * answers to.  Those are all the wood-into-charcoal recipe, so the Java
 * woods are put back by that rule and every other Bedrock name that Java
 * does not have is dropped.
 */
function furnaceRecipes() {
  const data = require('minecraft-data');
  const newest = data.versions.bedrock[0].minecraftVersion;
  const bedrock = data(`bedrock_${newest}`);
  const java = data(MINECRAFT_VERSION);
  const out = new Map();
  for (const recipe of Object.values(bedrock.recipes).flat()) {
    if (recipe.type !== 'furnace') continue;
    must(recipe.ingredients.length === 1,
      `furnace recipe ${recipe.name} takes ${recipe.ingredients.length}`);
    const input = recipe.ingredients[0].name;
    const output = recipe.output[0].name;
    if (java.itemsByName[input] === undefined) continue;
    if (java.itemsByName[output] === undefined) continue;
    out.set(input, output);
  }
  for (const name of Object.keys(java.itemsByName)) {
    if (!isWooden(name)) continue;
    if (!(name.endsWith('_log') || name.endsWith('_wood'))) continue;
    out.set(name, 'charcoal');
  }
  must(out.has('raw_iron') && out.has('oak_log') && out.has('sand'),
    `furnace recipes are missing staples: ${out.size} read`);
  return out;
}

const SMELTS_INTO = furnaceRecipes();

function parseArgs(argv) {
  const out = {
    seed: null,
    'mc-host': '127.0.0.1',
    'mc-port': '25565',
    'viewer-port': '3000',
    python: 'http://127.0.0.1:8000',
    username: 'jev',
  };
  for (let i = 0; i < argv.length; i += 2) {
    must(argv[i].startsWith('--'), `expected --option, got ${argv[i]}`);
    const key = argv[i].slice(2);
    must(key in out,
      `unknown option --${key}; known: ${Object.keys(out).join(' ')}`);
    must(argv[i + 1] !== undefined, `--${key} takes a value`);
    out[key] = argv[i + 1];
  }
  must(out.seed !== null,
    '--seed is required: it is the only record of which world this was');
  return out;
}

/* ------------------------------------------------------------------ server */

function post(base, path, body) {
  const url = new URL(path, base);
  const payload = Buffer.from(JSON.stringify(body));
  const agent = url.protocol === 'https:' ? https : http;
  return new Promise((resolve, reject) => {
    const req = agent.request(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': payload.length,
      },
    }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString();
        if (res.statusCode !== 200) {
          reject(new Bug(`POST ${url} -> ${res.statusCode} ${text}`));
          return;
        }
        resolve(JSON.parse(text));
      });
    });
    req.on('error', reject);
    req.end(payload);
  });
}

/* ------------------------------------------------------------ readout bits */

function compassOf(dx, dz) {
  const angle = Math.atan2(-dx, -dz) * 180 / Math.PI;
  const names = ['north', 'northeast', 'east', 'southeast',
    'south', 'southwest', 'west', 'northwest'];
  return names[(Math.round(angle / 45) + 8) % 8];
}

function bearing(from, to) {
  const dx = to.x - from.x;
  const dz = to.z - from.z;
  const dy = to.y - from.y;
  if (Math.hypot(dx, dz) < 2) {
    return dy >= 2 ? 'above' : (dy <= -2 ? 'below' : 'adjacent');
  }
  return compassOf(dx, dz);
}

function metres(from, to) {
  return `${Math.round(from.distanceTo(to))}m`;
}

/* Sky light is stored at its noon value, so it is dimmed by the clock the
 * way the game dims it; a block's own light is not. */
function lightLevel(block, isDay) {
  must(block !== null, 'no block at the bot head position');
  const sky = block.skyLight;
  const lamp = block.light;
  must(Number.isFinite(sky) && Number.isFinite(lamp),
    `block ${block.name} carries no light data: sky=${sky} light=${lamp}`);
  const level = Math.max(lamp, isDay ? sky : Math.max(0, sky - 11));
  if (level >= 15) return 'daylight';
  if (level >= 10) return 'bright';
  if (level >= 5) return 'dim';
  return 'dark';
}

function timePhrase(bot) {
  const tod = bot.time.timeOfDay;
  let phase;
  if (tod < 1000) phase = 'dawn';
  else if (tod < 6000) phase = 'morning';
  else if (tod < 9000) phase = 'midday';
  else if (tod < 12000) phase = 'afternoon';
  else if (tod < 13000) phase = 'dusk';
  else if (tod < 23000) phase = 'night';
  else phase = 'predawn';
  const toDusk = tod < 12000;
  const target = toDusk ? 12000 : (tod < 23000 ? 23000 : 24000);
  const minutes = Math.round((target - tod) / 1200);
  return `${phase}, ~${minutes} min to ${toDusk ? 'dusk' : 'dawn'}`;
}

function inventoryCounts(bot) {
  const out = {};
  for (const item of bot.inventory.items()) {
    out[item.name] = (out[item.name] || 0) + item.count;
  }
  return out;
}

/* Sorted by name so the same change always reads the same way. */
function countsDelta(before, after) {
  const parts = [];
  const names = [...new Set(
    [...Object.keys(before), ...Object.keys(after)])].sort();
  for (const name of names) {
    const change = (after[name] || 0) - (before[name] || 0);
    if (change !== 0) {
      parts.push(`${change > 0 ? '+' : ''}${change} ${name}`);
    }
  }
  return parts;
}

function gainedAny(before, after, names) {
  for (const name of names) {
    if ((after[name] || 0) > (before[name] || 0)) return true;
  }
  return false;
}

/* Every distinct block type within SCAN_RADIUS, nearest instance of each. */
function scanBlocks(bot) {
  const origin = bot.entity.position.floored();
  const byName = new Map();
  const cursor = new Vec3(0, 0, 0);
  for (let dy = -SCAN_HEIGHT; dy <= SCAN_HEIGHT; dy++) {
    for (let dx = -SCAN_RADIUS; dx <= SCAN_RADIUS; dx++) {
      for (let dz = -SCAN_RADIUS; dz <= SCAN_RADIUS; dz++) {
        const squared = dx * dx + dy * dy + dz * dz;
        if (squared > SCAN_RADIUS * SCAN_RADIUS) continue;
        cursor.set(origin.x + dx, origin.y + dy, origin.z + dz);
        const stateId = bot.world.getBlockStateId(cursor);
        if (stateId === undefined) continue;
        const block = bot.registry.blocksByStateId[stateId];
        if (block === undefined || AIR.includes(block.name)) continue;
        const held = byName.get(block.name);
        if (held === undefined || squared < held.squared) {
          byName.set(block.name, { squared, dx, dy, dz });
        }
      }
    }
  }
  const out = [];
  for (const [name, near] of byName) {
    const at = origin.offset(near.dx, near.dy, near.dz);
    const distance = Math.sqrt(near.squared);
    out.push({
      name,
      position: at,
      distance,
      entry: {
        type: name,
        nearest: `${Math.round(distance)}m ${bearing(origin, at)}`,
      },
    });
  }
  out.sort((a, b) => a.distance - b.distance);
  return out;
}

/* One ray from the bot's eyes to the entity's centre: the entity is
 * reported only when nothing solid stands in the way, so the bot's senses
 * stop where a player's would.  The server tracks entities through hills
 * and walls; this does not pass those on.
 *
 * A ray through unloaded chunks finds no block and so reads as clear, which
 * is absence of knowledge rather than a clear line, so an entity whose own
 * column is not loaded is not visible. */
function canSee(bot, entity) {
  must(Number.isFinite(entity.height),
    `entity ${entity.name} carries no height: ${entity.height}`);
  if (bot.world.getColumnAt(entity.position) === undefined) return false;
  const from = bot.entity.position.offset(0, EYE_HEIGHT, 0);
  const to = entity.position.offset(0, entity.height * 0.5, 0);
  const delta = to.minus(from);
  const distance = delta.norm();
  if (distance === 0) return true;
  const hit = bot.world.raycast(from, delta.scaled(1 / distance), distance);
  return hit === null || hit.intersect.distanceTo(from) >= distance;
}

/* The nearest VISIBLE_LIMIT entities the bot can actually see, grouped by
 * type with the nearest instance of each carrying the entry -- the same
 * rule the blocks use and the same rule the inventory uses for stacks.  A
 * school of thirty salmon is thirty entries otherwise, and the snapshot has
 * to stay a fixed size however crowded the water is.  Every distinct type
 * still appears; nothing is dropped for being uninteresting.
 *
 * Visibility is tested outward from the bot and the walk stops once the
 * limit is filled or SCAN_RADIUS is passed, so the cap never empties the
 * list when the closest entities all happen to be behind a wall, and the
 * raycasting stays bounded.  count is the number of visible instances of
 * that type, which is what the entries themselves describe. */
function nearbyEntities(bot) {
  const me = bot.entity;
  const candidates = [];
  for (const entity of Object.values(bot.entities)) {
    if (entity === me || entity.position === undefined) continue;
    if (entity.type === 'other' || entity.type === 'orb') continue;
    const distance = me.position.distanceTo(entity.position);
    if (distance > SCAN_RADIUS) continue;
    candidates.push({ entity, distance });
  }
  candidates.sort((a, b) => a.distance - b.distance);

  const visible = [];
  for (const near of candidates) {
    if (visible.length === VISIBLE_LIMIT) break;
    if (canSee(bot, near.entity)) visible.push(near);
  }

  const counts = new Map();
  for (const near of visible) {
    counts.set(near.entity.name, (counts.get(near.entity.name) || 0) + 1);
  }
  const seen = new Set();
  const out = [];
  for (const near of visible) {
    const type = near.entity.name;
    out.push({
      entity: near.entity,
      distance: near.distance,
      entry: seen.has(type) ? null : {
        type,
        count: counts.get(type),
        distance: metres(me.position, near.entity.position),
        direction: bearing(me.position, near.entity.position),
        hostile: near.entity.kind === 'Hostile mobs',
      },
    });
    seen.add(type);
  }
  return out;
}

/* The highest non-air block in a column, or null where the chunk is not
 * loaded. */
function surfaceAt(bot, x, z, aroundY) {
  if (bot.world.getColumnAt(new Vec3(x, 0, z)) === undefined) return null;
  const cursor = new Vec3(x, 0, z);
  for (let y = aroundY + 24; y >= aroundY - 32; y--) {
    cursor.y = y;
    const stateId = bot.world.getBlockStateId(cursor);
    if (stateId === undefined) return null;
    const block = bot.registry.blocksByStateId[stateId];
    if (block === undefined || AIR.includes(block.name)) continue;
    return { y, name: block.name, position: cursor.clone() };
  }
  return null;
}

function mode(names) {
  const counts = new Map();
  for (const name of names) counts.set(name, (counts.get(name) || 0) + 1);
  let best = null;
  for (const [name, n] of counts) {
    if (best === null || n > counts.get(best)) best = name;
  }
  return best;
}

function chunkKey(position) {
  return `${Math.floor(position.x / 16)},${Math.floor(position.z / 16)}`;
}

function directionsReport(bot, visited) {
  const here = bot.entity.position.floored();
  const hereSurface = surfaceAt(bot, here.x, here.z, here.y);
  const out = {};
  for (const [name, unit] of Object.entries(COMPASS)) {
    const tops = [];
    let farthest = null;
    for (const distance of PROBE_DISTANCES) {
      for (const offset of PROBE_SPREAD) {
        const x = here.x + unit.x * distance - unit.z * offset;
        const z = here.z + unit.z * distance + unit.x * offset;
        const top = surfaceAt(bot, x, z, here.y);
        if (top === null) continue;
        tops.push(top);
        if (farthest === null || distance >= farthest.distance) {
          farthest = { distance, top };
        }
      }
    }
    if (farthest === null) {
      out[name] = {
        biome: 'unloaded',
        height_change: 'unloaded',
        surface: 'unloaded',
        visited: false,
      };
      continue;
    }
    const biomeId = bot.world.getBiome(farthest.top.position);
    const biome = bot.registry.biomes[biomeId];
    must(biome !== undefined, `no biome for id ${biomeId}`);
    const base = hereSurface === null ? here.y : hereSurface.y;
    const change = farthest.top.y - base;
    out[name] = {
      biome: biome.name,
      height_change: `${change >= 0 ? '+' : ''}${change}m`,
      surface: `mostly ${mode(tops.map((t) => t.name))}`,
      visited: visited.has(chunkKey(farthest.top.position)),
    };
  }
  return out;
}

function readWorld(bot, visited, lastAction) {
  const head = bot.blockAt(bot.entity.position.offset(0, 1, 0));
  const floor = bot.blockAt(bot.entity.position.offset(0, -1, 0));
  const blocks = scanBlocks(bot);
  const entities = nearbyEntities(bot);
  const state = {
    self: {
      health: Math.round(bot.health),
      food: Math.round(bot.food),
      light: lightLevel(head, bot.time.isDay),
      standing_on: floor === null ? 'unloaded' : floor.name,
      /* Vertical position only.  X and Z are a global frame, which is what
       * a written route is made of; Y is a thing a player can feel. */
      y_level: Math.floor(bot.entity.position.y),
    },
    time: timePhrase(bot),
    holding: bot.heldItem === null ? 'nothing' : bot.heldItem.name,
    inventory: inventoryCounts(bot),
    blocks_in_range: blocks.map((b) => b.entry),
    entities: entities.filter((e) => e.entry !== null)
      .map((e) => e.entry),
    directions: directionsReport(bot, visited),
    last_action: lastAction,
  };
  return { state, blocks, entities };
}

/* ------------------------------------------------------------ milestones */

/* Observations of game state, nothing more.  No action is offered, ranked or
 * described because of anything below; the bot is never told these exist. */
function anyCount(counts, test) {
  let total = 0;
  for (const [name, count] of Object.entries(counts)) {
    if (test(name)) total += count;
  }
  return total;
}

const STONE_TOOLS = ['stone_pickaxe', 'stone_axe', 'stone_shovel',
  'stone_sword', 'stone_hoe'];

const MILESTONES = [
  {
    name: 'first_wood',
    test: (bot, counts) => anyCount(counts, (n) => n.endsWith('_log')) > 0,
  },
  {
    name: 'first_crafting_table',
    test: (bot, counts) => (counts.crafting_table || 0) > 0,
  },
  {
    name: 'first_stone_tool',
    test: (bot, counts) => STONE_TOOLS.some((n) => (counts[n] || 0) > 0),
  },
  {
    name: 'first_iron',
    test: (bot, counts) =>
      (counts.raw_iron || 0) + (counts.iron_ingot || 0) > 0,
  },
  {
    name: 'first_diamond',
    test: (bot, counts) => (counts.diamond || 0) > 0,
  },
  {
    /* A solid block directly above the bot's head while it is night: the
     * game's own definition of being under cover, read off the world. */
    name: 'first_sheltered_at_night',
    test: (bot) => {
      if (bot.time.isDay) return false;
      const over = bot.blockAt(bot.entity.position.offset(0, 2, 0));
      return over !== null && over.boundingBox === 'block';
    },
  },
  {
    name: 'survived_first_night',
    test: (bot) => bot.sawNight === true && bot.time.timeOfDay < 1000,
  },
];

function newMilestones(bot, counts, reached) {
  if (bot.time.timeOfDay >= 13000 && bot.time.timeOfDay < 23000) {
    bot.sawNight = true;
  }
  const out = [];
  for (const milestone of MILESTONES) {
    if (reached.has(milestone.name)) continue;
    if (!milestone.test(bot, counts)) continue;
    reached.add(milestone.name);
    out.push({ name: milestone.name, detail: null });
  }
  return out;
}

/* What the game requires to break a block type, read off the registry's
 * harvestTools: the items it accepts for that block.  This is a property of
 * the block, fixed for the type -- it does not read the inventory, the held
 * item, or anything about the action being described.  A block with no
 * harvestTools is one the game lets bare hands take. */
const TIER_LEVEL = {
  wooden: 0, golden: 0, stone: 1, iron: 2, diamond: 3, netherite: 3,
};
const LEVEL_TIER = { 1: 'stone', 2: 'iron', 3: 'diamond' };

function article(word) {
  return 'aeiou'.includes(word[0]) ? 'an' : 'a';
}

function toolRequirement(bot, name) {
  const block = bot.registry.blocksByName[name];
  must(block !== undefined, `no registry block named ${name}`);
  if (block.harvestTools === undefined) {
    return `${block.displayName} needs no tool.`;
  }
  const kinds = new Map();
  for (const id of Object.keys(block.harvestTools)) {
    const item = bot.registry.items[Number(id)];
    must(item !== undefined,
      `${name} lists item ${id}, which the registry has no name for`);
    const parts = item.name.split('_');
    const tiered = parts.length > 1 && TIER_LEVEL[parts[0]] !== undefined;
    const kind = tiered ? parts[parts.length - 1] : item.name;
    const level = tiered ? TIER_LEVEL[parts[0]] : null;
    const held = kinds.get(kind);
    if (held === undefined || (level !== null && held !== null
      && level < held)) {
      kinds.set(kind, level);
    }
  }
  const phrases = [];
  for (const [kind, level] of kinds) {
    if (level === null) {
      phrases.push(kind);
      continue;
    }
    if (level === 0) {
      phrases.push(`${article(kind)} ${kind}`);
      continue;
    }
    const tier = LEVEL_TIER[level];
    must(tier !== undefined, `${name} needs unknown tool level ${level}`);
    phrases.push(`${article(tier)} ${tier} ${kind} or better`);
  }
  return `${block.displayName} needs ${phrases.join(' or ')}.`;
}

/* ----------------------------------------------------------- action space */

/* Every action the game permits right now, and no others.  Nothing here asks
 * whether an action is a good idea, whether it fits a plan, or whether its
 * turn has come: craft_wooden_pickaxe is offered whenever the planks and
 * sticks exist and keeps being offered afterwards.  The list is not capped,
 * ranked or trimmed -- a long list is the correct output. */
function enumerateActions(bot, world) {
  const actions = [];
  const add = (label, description, run) => {
    actions.push({ label, description, run });
  };

  const movements = bot.pathfinder.movements;
  const held = bot.heldItem === null ? null : bot.heldItem.name;

  for (const block of world.blocks) {
    if (block.distance > GATHER_RANGE) break;
    const target = bot.blockAt(block.position);
    /* bot.canDigBlock is not the test here: it also asks whether the block
     * is within arm's reach of where the bot is standing now, and this
     * action walks there first. */
    if (target === null || !target.diggable) continue;
    /* Water and lava report themselves as diggable with the hardness the
     * game gives fluids; nothing breaks them.  The set is mineflayer's
     * own. */
    if (movements.liquids.has(target.type)) continue;
    /* Bare hands on obsidian is a real dig that simply outlasts the clock
     * this action is given, so it can never finish. */
    const digMs = bot.digTime(target);
    if (!(digMs < ACTION_TIMEOUT_MS)) continue;
    add(`break_${block.name}`,
      `Walk to the ${block.name} ${block.entry.nearest} and break it. `
      + toolRequirement(bot, block.name),
      () => gather(bot, block));
  }

  const counts = inventoryCounts(bot);

  /* A fluid is not broken, it is scooped, and only into an empty bucket and
   * only out of a source block: a flowing one gives nothing. */
  if ((counts.bucket || 0) > 0) {
    for (const block of world.blocks) {
      if (block.distance > GATHER_RANGE) break;
      if (block.name !== 'water' && block.name !== 'lava') continue;
      const target = bot.blockAt(block.position);
      if (target === null) continue;
      const level = target.getProperties().level;
      must(level !== undefined, `${block.name} carries no level property`);
      if (Number(level) !== 0) continue;
      add(`fill_bucket_${block.name}`,
        `Walk to the ${block.name} ${block.entry.nearest} and fill a bucket `
        + `from it, giving a ${block.name}_bucket.`,
        () => fillBucket(bot, block));
    }
  }

  const table = bot.findBlock({
    matching: bot.registry.blocksByName.crafting_table.id,
    maxDistance: 4,
  });
  for (const id of Object.keys(bot.registry.recipes)) {
    const recipes = bot.recipesFor(Number(id), null, 1, table);
    if (recipes.length === 0) continue;
    const item = bot.registry.items[Number(id)];
    if (item === undefined) continue;
    const needs = recipes[0].delta
      .filter((d) => d.count < 0)
      .map((d) => `${-d.count} ${bot.registry.items[d.id].name}`)
      .join(', ');
    add(`craft_${item.name}`,
      `Craft ${item.name}. You have the ${needs} needed`
      + `${table === null ? '' : ', and a crafting table is in reach'}.`,
      () => bot.craft(recipes[0], 1, table));
  }

  /* A furnace needs the block itself in reach, something the game smelts,
   * and something else in the inventory to burn under it. */
  const furnace = bot.findBlock({
    matching: bot.registry.blocksByName.furnace.id,
    maxDistance: 4,
  });
  if (furnace !== null) {
    const fuel = Object.keys(counts).find(
      (name) => isFuel(name) && SMELTS_INTO.get(name) === undefined);
    for (const name of Object.keys(counts)) {
      const output = SMELTS_INTO.get(name);
      if (output === undefined) continue;
      const burning = fuel === undefined && isFuel(name) && counts[name] > 1
        ? name : fuel;
      if (burning === undefined) continue;
      add(`smelt_${name}`,
        `Smelt the ${name} in the furnace in reach, burning ${burning}, `
        + `giving ${output}.`,
        () => smelt(bot, name, burning));
    }
  }

  for (const name of Object.keys(counts)) {
    if (name === held) continue;
    add(`equip_${name}`, `Hold the ${name}.`, () => equip(bot, name));
  }

  for (const name of Object.keys(counts)) {
    if (bot.registry.foodsByName[name] === undefined) continue;
    /* A full hunger bar refuses ordinary food and takes the few the game
     * marks as always edible. */
    if (bot.food >= 20 && !ALWAYS_EDIBLE.has(name)) continue;
    add(`eat_${name}`, `Eat ${name}. Food is ${bot.food}/20.`,
      () => eat(bot, name));
  }

  for (const mob of world.entities) {
    if (mob.distance > MOB_RANGE) break;
    /* One pair per distinct type, aimed at the nearest of that type; the
     * duplicates behind it are the same action. */
    if (mob.entry === null) continue;
    if (!['mob', 'hostile', 'animal', 'player'].includes(mob.entity.type)) {
      continue;
    }
    /* Gone between the readout and here: the swing would land on nothing. */
    if (!mob.entity.isValid) continue;
    add(`attack_${mob.entry.type}`,
      `Attack the ${mob.entry.type} ${mob.entry.distance} `
      + `${mob.entry.direction}.`,
      () => attack(bot, mob.entity));
    /* Fleeing is offered for the mobs the game itself categorizes as
     * hostile.  Attacking stays offered for every mob: it is how the bot
     * eats. */
    if (mob.entity.kind === 'Hostile mobs') {
      add(`flee_${mob.entry.type}`,
        `Move away from the ${mob.entry.type} ${mob.entry.distance} `
        + `${mob.entry.direction}.`,
        () => flee(bot, mob.entity));
    }
  }

  for (const [name, unit] of Object.entries(COMPASS)) {
    const report = world.state.directions[name];
    add(`explore_${name}`,
      `Head ${name}, toward ${report.biome}. `
      + `${report.visited ? 'Visited' : 'Unvisited'}.`,
      () => explore(bot, unit));
  }

  /* Digging, one square at a time, in a direction the model names.  What is
   * on the other side of the block is not read and not reported: these are
   * blind, and the descriptions say so. */
  const under = bot.entity.onGround
    ? digCandidate(bot, movements,
      bot.entity.position.floored().offset(0, -1, 0))
    : null;
  if (under !== null) {
    add('dig_down', `Break the ${under.name} directly beneath you and drop `
      + 'into the space it leaves. You cannot see what is under it.',
      () => digDown(bot));
  }

  const over = digCandidate(bot, movements,
    bot.entity.position.floored().offset(0, 2, 0));
  if (over !== null) {
    add('dig_up', `Break the ${over.name} directly above your head. This `
      + 'opens the ceiling; it does not move you, and you stay where you '
      + 'are.',
      () => digUp(bot));
  }

  const footing = held !== null && bot.registry.blocksByName[held] !== undefined
    ? pillarFooting(bot, movements) : null;
  if (footing !== null) {
    add('pillar_up', `Jump and place the ${held} you are holding underneath `
      + 'yourself, leaving you standing one block higher.',
      () => pillarUp(bot));
  }

  const LEVEL = { 0: 'at your feet', 1: 'at your head' };
  for (const [name, unit] of Object.entries(COMPASS)) {
    const targets = digTargets(bot, movements, unit);
    if (targets.length === 0) continue;
    const list = targets.map(
      (t) => `the ${t.name} ${LEVEL[t.dy]}`).join(' and ');
    add(`dig_${name}`,
      `Break ${list} one square ${name} of you, then step into the gap. `
      + 'You cannot see what is behind it.',
      () => digToward(bot, unit, targets));
  }

  const spot = held === null || bot.registry.blocksByName[held] === undefined
    ? null : placeSpot(bot, movements);
  if (spot !== null) {
    add('place_held', `Place the ${held} you are holding on the ground `
      + 'ahead.', () => placeHeld(bot, spot));
  }

  add('wait', 'Do nothing this tick.', () => sleep(1000));
  return actions;
}

/* --------------------------------------------------------------- executing */

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* spawn only means the server has told us where we are; the chunk we are in
 * arrives after it, and again after every respawn.  Reading the world before
 * it is there reads an empty world, so every tick starts here. */
async function waitForWorld(bot) {
  const deadline = Date.now() + WORLD_WAIT_MS;
  for (;;) {
    const head = bot.blockAt(bot.entity.position.offset(0, 1, 0));
    if (head !== null) return;
    must(Date.now() < deadline,
      `no chunk at ${bot.entity.position} ${WORLD_WAIT_MS}ms after spawn`);
    await sleep(100);
  }
}

function itemNamed(bot, name) {
  const item = bot.inventory.items().find((i) => i.name === name);
  if (item === undefined) throw new Error(`no ${name} in the inventory`);
  return item;
}

/* Item entities lying within DROP_RADIUS of a point, by entity id. */
function dropsNear(bot, position) {
  const out = new Map();
  for (const entity of Object.values(bot.entities)) {
    if (entity.name !== 'item' || entity.position === undefined) continue;
    if (entity.position.distanceTo(position) > DROP_RADIUS) continue;
    out.set(entity.id, entity);
  }
  return out;
}

async function until(test, ms) {
  const deadline = Date.now() + ms;
  for (;;) {
    if (test()) return true;
    if (Date.now() >= deadline) return false;
    await sleep(POLL_MS);
  }
}

/* Walk onto the drop a break left and let the game hand it over.  A drop
 * that cannot be reached, burns up or floats away is an ordinary thing for
 * the world to do, so nothing here throws: the caller says what happened.
 *
 * Reports only what it watched: whether a new drop appeared at the broken
 * block, and whether that drop is gone from the world by the end. */
async function collect(bot, position, before) {
  const appeared = await until(
    () => dropsNear(bot, position).size > before.size, DROP_SPAWN_MS);
  if (!appeared) return 'none';
  const drops = dropsNear(bot, position);
  for (const id of before.keys()) drops.delete(id);
  let nearest = null;
  for (const entity of drops.values()) {
    const distance = bot.entity.position.distanceTo(entity.position);
    if (nearest === null || distance < nearest.distance) {
      nearest = { entity, distance };
    }
  }
  must(nearest !== null,
    `a drop appeared near ${position} and then none was there`);
  const deadline = Date.now() + COLLECT_MS;
  const walk = bot.pathfinder.goto(new goals.GoalNear(
    nearest.entity.position.x, nearest.entity.position.y,
    nearest.entity.position.z, 1)).then(() => true, () => false);
  const arrived = await Promise.race(
    [walk, sleep(COLLECT_MS).then(() => false)]);
  if (!arrived) bot.pathfinder.stop();
  await until(() => !nearest.entity.isValid,
    Math.max(POLL_MS, deadline - Date.now()));
  return nearest.entity.isValid ? 'left' : 'taken';
}

/* What the game says this block turns into when it is broken like this. */
function dropNames(bot, block, heldType) {
  if (!block.canHarvest(heldType)) return [];
  const out = [];
  for (const drop of block.drops) {
    const id = typeof drop === 'object' ? drop.drop : drop;
    const item = bot.registry.items[
      typeof id === 'object' ? id.id : id];
    if (item !== undefined) out.push(item.name);
  }
  return out;
}

/* Break a block that is already in reach and say what became of its drop.
 *
 * A note names the block it is about, and no note claims an absence while
 * anything at all arrived in the inventory: the drop is recognised by the
 * item the game says this block yields, and when something else turned up
 * instead -- an item walked over on the way to the drop -- there is nothing
 * this can honestly say, so it says nothing and the delta stands alone. */
async function breakAndCollect(bot, target) {
  const name = target.name;
  const position = target.position;
  const wanted = dropNames(bot, target, bot.heldItem?.type ?? null);
  const held = dropsNear(bot, position);
  const before = inventoryCounts(bot);
  await bot.dig(target);
  const fate = await collect(bot, position, held);
  const after = inventoryCounts(bot);
  if (gainedAny(before, after, wanted)) return null;
  if (countsDelta(before, after).some((p) => p.startsWith('+'))) return null;
  if (fate === 'taken') return `the ${name} drop did not reach the inventory`;
  if (fate === 'left') return `the ${name} drop was not picked up`;
  return `no ${name} drop appeared`;
}

async function gather(bot, block) {
  await bot.pathfinder.goto(
    new goals.GoalLookAtBlock(block.position, bot.world, { reach: 4 }));
  const target = bot.blockAt(block.position);
  if (target === null || target.name !== block.name) {
    throw new Error(`the ${block.name} is no longer there`);
  }
  return breakAndCollect(bot, target);
}

/* One block, straight down, and then whatever is under it.  Nothing here
 * looks before the bot drops. */
async function digDown(bot) {
  const startY = bot.entity.position.y;
  const target = bot.blockAt(bot.entity.position.floored().offset(0, -1, 0));
  if (target === null || AIR.includes(target.name)) {
    throw new Error('there is nothing underfoot to break');
  }
  const note = await breakAndCollect(bot, target);
  await until(() => bot.entity.position.y <= startY - 0.9
    && bot.entity.onGround, DROP_SPAWN_MS);
  return note;
}

/* One block out of the ceiling.  The bot does not move. */
async function digUp(bot) {
  const target = bot.blockAt(bot.entity.position.floored().offset(0, 2, 0));
  if (target === null || AIR.includes(target.name)) {
    throw new Error('there is nothing overhead to break');
  }
  return breakAndCollect(bot, target);
}

/* A block the game will let this bot break within one action's clock, or
 * null.  The same standard the gather gate applies, at a named place. */
function digCandidate(bot, movements, at) {
  const block = bot.blockAt(at);
  if (block === null || AIR.includes(block.name)) return null;
  if (!block.diggable) return null;
  if (movements.liquids.has(block.type)) return null;
  if (!(bot.digTime(block) < ACTION_TIMEOUT_MS)) return null;
  return block;
}

/* The two squares a body occupies, one step away along unit, when the pair
 * of them can be broken inside one action's clock.  Both go in one action,
 * so it is the pair that has to fit, not each block on its own. */
function digTargets(bot, movements, unit) {
  const base = bot.entity.position.floored();
  const out = [];
  let digMs = 0;
  for (const dy of [0, 1]) {
    const at = base.offset(unit.x, dy, unit.z);
    const block = digCandidate(bot, movements, at);
    if (block === null) continue;
    digMs += bot.digTime(block);
    out.push({ dy, at, name: block.name });
  }
  return digMs < ACTION_TIMEOUT_MS ? out : [];
}

/* Walk forward until the bot is in the column it just opened.  A step, not
 * a route: no goal, no search, and it may simply not get there. */
async function stepInto(bot, at) {
  await bot.lookAt(at.offset(0.5, EYE_HEIGHT, 0.5), true);
  bot.setControlState('forward', true);
  await until(() => {
    const here = bot.entity.position.floored();
    return here.x === at.x && here.z === at.z;
  }, STEP_MS);
  bot.setControlState('forward', false);
}

async function digToward(bot, unit, targets) {
  const notes = new Set();
  for (const target of targets) {
    const block = bot.blockAt(target.at);
    if (block === null || AIR.includes(block.name)) continue;
    const note = await breakAndCollect(bot, block);
    if (note !== null) notes.add(note);
  }
  const base = bot.entity.position.floored();
  await stepInto(bot, base.offset(unit.x, 0, unit.z));
  return notes.size === 0 ? null : [...notes].join('; ');
}

async function fillBucket(bot, block) {
  await bot.pathfinder.goto(
    new goals.GoalLookAtBlock(block.position, bot.world, { reach: 3 }));
  const target = bot.blockAt(block.position);
  if (target === null || target.name !== block.name) {
    throw new Error(`the ${block.name} is no longer there`);
  }
  await bot.equip(itemNamed(bot, 'bucket'), 'hand');
  await bot.lookAt(block.position.offset(0.5, 0.5, 0.5), true);
  bot.activateItem();
  await sleep(500);
  const filled = `${block.name}_bucket`;
  if (!bot.inventory.items().some((i) => i.name === filled)) {
    throw new Error(`the ${block.name} did not go into the bucket`);
  }
}

/* One item in, one out: the furnace is opened, loaded, watched until the
 * output appears and emptied again, inside the one action. */
async function smelt(bot, inputName, fuelName) {
  const block = bot.findBlock({
    matching: bot.registry.blocksByName.furnace.id,
    maxDistance: 4,
  });
  if (block === null) throw new Error('the furnace is no longer in reach');
  const furnace = await bot.openFurnace(block);
  try {
    if (furnace.fuelItem() === null) {
      await furnace.putFuel(itemNamed(bot, fuelName).type, null, 1);
    }
    await furnace.putInput(itemNamed(bot, inputName).type, null, 1);
    const deadline = Date.now() + SMELT_WAIT_MS;
    for (;;) {
      if (furnace.outputItem() !== null) break;
      if (Date.now() >= deadline) {
        throw new Error('the furnace did not melt it in time');
      }
      await sleep(200);
    }
    await furnace.takeOutput();
  } finally {
    furnace.close();
  }
}

async function equip(bot, name) {
  await bot.equip(itemNamed(bot, name), 'hand');
}

async function eat(bot, name) {
  await bot.equip(itemNamed(bot, name), 'hand');
  await bot.consume();
}

async function attack(bot, entity) {
  await bot.pathfinder.goto(new goals.GoalFollow(entity, 2));
  if (!entity.isValid) throw new Error('it is gone');
  await bot.lookAt(entity.position.offset(0, entity.height * 0.5, 0), true);
  bot.attack(entity);
  await sleep(600);
}

async function flee(bot, entity) {
  const away = bot.entity.position.minus(entity.position);
  const scale = 12 / Math.max(1, Math.hypot(away.x, away.z));
  await bot.pathfinder.goto(new goals.GoalNearXZ(
    Math.round(bot.entity.position.x + away.x * scale),
    Math.round(bot.entity.position.z + away.z * scale),
    3,
  ));
}

async function explore(bot, unit) {
  await bot.pathfinder.goto(new goals.GoalNearXZ(
    Math.round(bot.entity.position.x + unit.x * EXPLORE_DISTANCE),
    Math.round(bot.entity.position.z + unit.z * EXPLORE_DISTANCE),
    3,
  ));
}

/* Where the held block would go: the square the bot faces, which has to be
 * free, standing on something solid to place against.  Found when the action
 * is offered, not when it runs, because whether such a square exists is the
 * whole question of whether placing is possible at all. */
function placeSpot(bot, movements) {
  const yaw = bot.entity.yaw;
  const ahead = bot.entity.position.floored()
    .offset(Math.round(-Math.sin(yaw)), 0, Math.round(-Math.cos(yaw)));
  const target = bot.blockAt(ahead);
  /* Nothing may already occupy the square.  emptyBlocks is mineflayer's set
   * of blocks with no collision shape at all -- air, water, grass, snow --
   * which is wider than what the game lets a block replace, and the wider
   * side is the right one to err on. */
  if (target === null || !movements.emptyBlocks.has(target.type)) return null;
  const reference = bot.blockAt(ahead.offset(0, -1, 0));
  if (reference === null || reference.boundingBox !== 'block') return null;
  return ahead;
}

async function placeHeld(bot, ahead) {
  if (bot.heldItem === null) throw new Error('holding nothing');
  const reference = bot.blockAt(ahead.offset(0, -1, 0));
  if (reference === null || reference.boundingBox !== 'block') {
    throw new Error('nothing solid to place against');
  }
  await bot.lookAt(ahead, true);
  await bot.placeBlock(reference, new Vec3(0, 1, 0));
}

/* The block the pillar would be placed against -- the one underfoot, whose
 * top face the new block goes on -- or null where the game would not let the
 * bot rise a block from here: off the ground, nothing solid to place against,
 * or something occupying the two squares the jump and the body need. */
function pillarFooting(bot, movements) {
  if (!bot.entity.onGround) return null;
  const base = bot.entity.position.floored();
  const reference = bot.blockAt(base.offset(0, -1, 0));
  if (reference === null || reference.boundingBox !== 'block') return null;
  for (const dy of [1, 2]) {
    const block = bot.blockAt(base.offset(0, dy, 0));
    if (block === null || !movements.emptyBlocks.has(block.type)) return null;
  }
  return reference;
}

/* Jump, and put the held block on the top face of the block underfoot while
 * the body is clear of that square.  A mistimed jump or a refusal leaves the
 * bot where it was, which is a thing the world does and not a fault; what
 * left the inventory is reconciled in run() like any other placement. */
async function pillarUp(bot) {
  const base = bot.entity.position.floored();
  const startY = base.y;
  const reference = bot.blockAt(base.offset(0, -1, 0));
  if (reference === null || reference.boundingBox !== 'block') {
    throw new Error('nothing solid underfoot to place against');
  }
  const name = bot.heldItem === null ? 'block' : bot.heldItem.name;
  await bot.lookAt(reference.position.offset(0.5, 1, 0.5), true);
  bot.setControlState('jump', true);
  const airborne = await until(
    () => bot.entity.position.y >= startY + 1, JUMP_MS);
  bot.setControlState('jump', false);
  if (airborne) {
    try {
      await bot._placeBlockWithOptions(reference, new Vec3(0, 1, 0),
        { swingArm: 'right', forceLook: 'ignore' });
    } catch (error) {
      if (error instanceof Bug) throw error;
    }
  }
  const risen = await until(() => bot.entity.onGround
    && bot.entity.position.floored().y === startY + 1, SETTLE_MS);
  return risen ? null : `the ${name} did not go down underfoot`;
}

/* Abandoning a promise does not stop the body, so everything the executors
 * start is stopped here by hand. */
async function halt(bot) {
  /* pathfinder.stop() only raises a flag, which the next path it is given
   * consumes and dies on.  Raising it while nothing is pathing therefore
   * kills the following action instead of this one. */
  if (bot.pathfinder.isMoving() || bot.pathfinder.isMining()
    || bot.pathfinder.isBuilding()) {
    bot.pathfinder.stop();
  }
  bot.clearControlStates();
  if (bot.targetDigBlock) bot.stopDigging();
  await sleep(POLL_MS);
}

const INTERRUPTED = Symbol('interrupted');

/* A game outcome comes back as a result string and the loop continues; a
 * fault in this program is a Bug and takes the process down.
 *
 * Damage, and only damage, cuts an action short: the loop then asks for a
 * decision against the situation that did the damage instead of finishing a
 * walk that is no longer the question.  Nothing else interrupts -- what to
 * do about a mob in view, the dark or an empty stomach is the model's to
 * decide, on its own decision. */
/* The pathfinder's own rejections, in words.  With nothing to build from,
 * a target the bot cannot walk to is an ordinary state of the world. */
const WALK_FAILURES = new Map([
  ['NoPath', 'no walkable route to it'],
  ['Timeout', 'no walkable route found in time'],
  ['PathStopped', 'the walk stopped short of it'],
  ['GoalChanged', 'the walk was redirected'],
]);

async function run(action, bot) {
  const before = inventoryCounts(bot);
  const beforeHealth = Math.round(bot.health);
  const startHealth = bot.health;
  bot.placed.length = 0;
  let signal;
  const damaged = new Promise((resolve) => { signal = resolve; });
  const watch = () => {
    if (bot.health < startHealth) signal();
  };
  bot.on('health', watch);
  let outcome;
  try {
    /* An executor may return a word on what the world did with it -- a dig
     * whose drop never reached the inventory did happen, and saying so is
     * not a failure. */
    const note = await Promise.race([
      action.run(),
      damaged.then(() => INTERRUPTED),
      sleep(ACTION_TIMEOUT_MS).then(() => {
        throw new Error('timed out');
      }),
    ]);
    if (note === INTERRUPTED) {
      await halt(bot);
      outcome = 'interrupted: took damage';
    } else {
      outcome = typeof note === 'string' ? `ok, ${note}` : 'ok';
    }
  } catch (error) {
    if (error instanceof Bug) throw error;
    await halt(bot);
    const plain = WALK_FAILURES.get(error.name);
    outcome = `failed: ${plain === undefined ? error.message : plain}`;
  } finally {
    bot.removeListener('health', watch);
  }
  const after = inventoryCounts(bot);
  const parts = countsDelta(before, after);
  /* Where an otherwise unexplained loss went: a block the bot put down,
   * counted only as far as the inventory actually fell. */
  const placed = {};
  for (const name of bot.placed) placed[name] = (placed[name] || 0) + 1;
  for (const name of Object.keys(placed).sort()) {
    const lost = (before[name] || 0) - (after[name] || 0);
    const n = Math.min(placed[name], lost);
    if (n > 0) parts.push(`placed ${n} ${name}`);
  }
  const lostHealth = beforeHealth - Math.round(bot.health);
  if (lostHealth > 0) parts.push(`-${lostHealth} health`);
  return `${outcome}${parts.length ? `, ${parts.join(', ')}` : ''}`;
}

/* -------------------------------------------------------------------- loop */

function watchForDeath(bot, onDeath) {
  const ChatMessage = require('prismarine-chat')(bot.registry);
  let cause = null;
  bot._client.on('death_combat_event', (packet) => {
    cause = ChatMessage.fromNotch(packet.message).toString();
  });
  bot.on('death', () => {
    onDeath(cause === null ? 'the game gave no cause' : cause);
    cause = null;
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const mcHost = args['mc-host'];
  const mcPort = Number(args['mc-port']);
  const viewerPort = Number(args['viewer-port']);
  must(Number.isInteger(mcPort),
    `--mc-port must be a number, got ${args['mc-port']}`);
  must(Number.isInteger(viewerPort),
    `--viewer-port must be a number, got ${args['viewer-port']}`);

  console.log(
    `connecting to Minecraft at ${mcHost}:${mcPort} as ${args.username}`);
  console.log(`world seed ${args.seed}`);
  console.log(`viewer on http://127.0.0.1:${viewerPort}/`);
  console.log(`decisions from ${args.python}`);

  const bot = mineflayer.createBot({
    host: mcHost,
    port: mcPort,
    username: args.username,
    version: MINECRAFT_VERSION,
    auth: 'offline',
  });
  bot.loadPlugin(pathfinder);

  const visited = new Set();
  const reached = new Set();
  /* Everything that has happened to this bot, from the first decision to
   * this one, sent whole with every decision.  The Python server keeps no
   * authoritative copy of it, so restarting it mid-run loses nothing. */
  const history = [];
  let decisions = 0;
  const remember = (name, detail) => {
    history.push({ name, detail, decision: decisions + 1,
      ticks: bot.time.age });
  };

  bot.on('kicked', (reason) => {
    throw new Bug(`kicked: ${reason}`);
  });
  bot.on('error', (error) => {
    throw error;
  });

  await new Promise((resolve) => bot.once('spawn', resolve));
  await waitForWorld(bot);
  watchForDeath(bot, (cause) => {
    remember('death', cause);
    console.log(`died: ${cause}`);
  });
  const movements = new Movements(bot);
  /* The pathfinder will otherwise spend the bot's own dirt and cobblestone
   * to build towers up and bridges across, while walking to somewhere the
   * model asked to go: a resource decision the model did not take and was
   * not told about.  Emptying the scaffolding list leaves every route with
   * no blocks to place, which is what both flags below read; pillar_up is
   * where a block goes down deliberately. */
  movements.allow1by1towers = false;
  movements.scafoldingBlocks = [];
  bot.pathfinder.setMovements(movements);

  /* What the bot tried to put down: blockPlaced does not fire for every
   * placement -- mineflayer cannot always tell the server's answer apart,
   * and throws where it cannot.  The attempt is recorded here and
   * reconciled against the inventory in run(), so nothing is claimed that
   * the inventory does not show. */
  bot.placed = [];
  const genericPlace = bot._genericPlace.bind(bot);
  bot._genericPlace = (reference, face, options) => {
    if (bot.heldItem !== null) bot.placed.push(bot.heldItem.name);
    return genericPlace(reference, face, options);
  };
  /* Third person: in first person the camera is the bot's aim, which the
   * pathfinder and every dig snap around about once a second. */
  mineflayerViewer(bot, {
    port: viewerPort,
    firstPerson: false,
    viewDistance: VIEWER_VIEW_DISTANCE,
  });

  const session = {
    seed: args.seed,
    viewer: `http://127.0.0.1:${viewerPort}/`,
    minecraft: `${mcHost}:${mcPort}`,
    version: MINECRAFT_VERSION,
  };
  await post(args.python, '/minecraft/api/session', session);

  let lastAction = { did: 'spawn', result: 'ok' };
  for (;;) {
    const started = Date.now();
    await waitForWorld(bot);
    visited.add(chunkKey(bot.entity.position));
    for (const reachedNow of newMilestones(bot, inventoryCounts(bot), reached)) {
      remember(reachedNow.name, reachedNow.detail);
    }

    const world = readWorld(bot, visited, lastAction);
    const actions = enumerateActions(bot, world);
    decisions += 1;

    const answer = await post(args.python, '/minecraft/api/decide', {
      session,
      decision: decisions,
      state: world.state,
      actions: actions.map(
        (a) => ({ label: a.label, description: a.description })),
      history,
      ticks: bot.time.age,
    });
    const action = actions.find((a) => a.label === answer.chosen);
    must(action !== undefined,
      `the server chose ${answer.chosen}, which was not offered`);

    console.log(
      `${answer.objective} | ${action.label} (of ${actions.length})`);
    const result = await run(action, bot);
    console.log(`  ${result}`);
    lastAction = { did: action.label, result };

    const elapsed = Date.now() - started;
    if (elapsed < TICK_MS) await sleep(TICK_MS - elapsed);
  }
}

main();
