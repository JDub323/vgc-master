# Code-review data contracts

This is the reviewer-facing companion to the inline docstrings. It names the
plain-dict, tuple, NumPy, and Torch structures passed between modules, then
catalogs every production function/method by input and output. Runtime
``TypedDict``/alias definitions live in [contracts.py](contracts.py); they are
documentation only and do not wrap or copy the existing values.

## Notation and ownership

- `Set`: `contracts.PokemonSet`; names are Showdown ids unless stated.
- `State`: `contracts.PositionState`, the CTS view from `LogParser._view`.
- `Request`: `contracts.ShowdownRequest`, a partial raw Showdown request.
- `JointAction`: `(SlotAction, SlotAction)`; `JointIndex` is its flattened
  `0..1520` model label.
- `ChoiceInfo`: diagnostics returned with a chosen joint action.
- `Belief`: an externally owned, per-battle `BeliefModel`. A chooser consumes
  it but never creates or updates the live belief lifecycle.
- `tokens`: NumPy `uint16[n_tokens]` outside Torch; training converts them to
  Torch `int64[B,n_tokens]`.
- `joint_dists`: NumPy float `[B,1521]`, normalized across statically valid
  joint actions. `values` is NumPy float `[B]` in `[-1,1]`.
- Functions named `close`, `destroy`, `update`, `feed`, `run`, `save`, or
  `write` mutate/perform I/O and return `None` unless their row says otherwise.
- Constructors consume the listed values, initialize mutable object state, and
  return the new instance through normal Python construction.
- A catalog row written as ``ClassName(...)`` is the contract for that class's
  ``ClassName.__init__`` method.
- Nested CLI callbacks (`worker`, `flush`, HTTP handlers, etc.) use the
  enclosing function's captured structures and are documented with that
  enclosing function rather than as public APIs.

## Cross-module structures

### Pokémon sets

`PokemonSet` is a mapping with:

```text
name/species/item/ability/nature/gender: str
moves: list[str]                 # up to four ids
evs: list[int]                   # six Champions stat-point values
level: int
```

Belief particles use `moves: tuple[str,...]`, optional `evs`, posterior-only
`arch`, and frequency `n`. `env.full_set` converts either shape to the full
`PokemonSet` required by `Side` and `pack_team`.

### CTS position state

`LogParser._view(side_id)` returns:

```text
{
  turn: int, weather: str, terrain: str, trickroom: bool,
  my:  {team: list[MonOwnView], mega_available: bool,
        conditions: dict[str,bool]},
  opp: {team: list[MonOpponentView], mega_available: bool,
        conditions: dict[str,bool]}
}
```

Own mon views contain the full `set`; opponent views contain only public
species/HP/status/boosts/slot plus `revealed_moves`, `revealed_item`,
`revealed_ability`, and consumption/mega/Protect counters.

### Belief events and outputs

Events are discriminated tuples:

```text
('reveal', side_id, team_idx, 'move'|'item'|'ability', name)
('consumed', side_id, team_idx, item)
('mega', side_id, team_idx)
('move_order', list[(side_id, team_idx, move_id, speed_context)], {'tr': bool})
('dmg', atk_side, atk_idx, move, def_side, def_idx, hp_fraction, damage_context)
```

`speed_context` is `{spe:int, par:bool, tw:bool}`. `damage_context` carries
`crit/spread/multi/burn`, weather/terrain/screens, attacker/defender boosts,
fainted-allies count, defender HP before the hit, and transformation state.

`Belief.summary()` returns `dict[team_idx, BeliefSummaryEntry]`; each entry has
modal item/nature/archetype and their masses, 5%-95% speed bounds, and expected
bulk. `sample_sets(n,rng)` returns `list[n][team_idx]` full sampled set dicts.

### Actions and chooser output

`SlotAction(kind, move_slot, target, mega, switch_to)` uses `kind` in
`pass|move|switch`. `MoveChooser.choose(...)` returns:

```text
(
  (slot_a_action, slot_b_action),
  {
    value: float, solve: bool,
    visits: list[(flat_joint_index, unnormalized_count)],
    strategy: list[(human_description, probability)],
    q: list[(human_description, mean_value)],
    opp_pred: list[(human_description, probability)],
    health: dict[str, float]
  }
)
```

Simple baselines omit `visits` because they are not self-play search targets.

### Search objects

A `DetGame` owns one sampled opponent team, reconstructed root
`SidecarBattle`, tracker, collapsed beliefs, root `Node`, and serialized root
state. A `Node` owns per-player action lists and parallel NumPy arrays:
`my_p/my_n/my_w`, `opp_p/opp_n/opp_w`, scalar `n/value`, and
`children[(my_index,opp_index)] -> Node`. Root aggregation rows are mutable
`[JointAction, count, value_sum]` lists.

### Archives and benchmark results

`agent.json` is `AgentSpecJson`: schema version, top-level implementation ID,
architecture label, config path, `name -> BrickSpecJson`, behavior assets,
runtime identities, source hashes, and archive metadata. All behavior paths
are archive-relative and traversal-checked.

One benchmark registry result is `BenchmarkResult`: contestant/team names,
`winner: a|b|tie`, turns/search overrides/date, archive/run era IDs, source
commit, and per-side agent implementation/architecture IDs.

### Training and self-play batches

`LogParser.parse()` returns a `ParsedBattle` with battle identity, players,
ratings, two oracle teams, winner, and `list[ParsedTurn]`. Each nonzero turn
contains `states: dict[SideID,PositionState]`,
`actions: dict[SideID,tuple[int,int]|None]`, and `list[BeliefEvent]`; synthetic
turn zero carries preview events and has `states/actions = None`. `data.parse`
adds `split: train|val|test` before pickling the record.

A BC NPZ shard (`BCShard`) contains:

```text
tokens:     uint16[N,T]       acts:       int8[N,2]
value:      int8[N]           weight:     float32[N]
opp_items:  int16[N,6]        opp_abils:  int16[N,6]
opp_moves:  int16[N,6,4]      dmg_active: uint8[N,2,4,2]
```

BC dataset `__getitem__` converts the first seven fields (not the offline-only
`dmg_active` baseline feature) to seven Torch tensors:

```text
tokens[B,T], acts[B,2], value[B], weight[B],
opp_items[B,6], opp_abilities[B,6], opp_moves[B,6,4]
```

Self-play NPZ shards retain `acts/weight`, replace `dmg_active` with sparse
policy targets `pol_idx:int16[N,K]`, `pol_p:float32[N,K]`, and otherwise use
the same dtypes/shapes. In-memory decisions are `SelfPlaySample` mappings;
generation attaches outcome and opponent-oracle labels while flushing them.

## Function contract catalog

### `actions.py`

| function | input | output |
|---|---|---|
| `SlotAction(...)` | `kind:str`; move/target/mega/switch integer fields | Immutable action value object. |
| `to_index(a)` | `SlotAction` | Slot index `int` in `0..38`. |
| `from_index(i)` | Slot index `int` | Reconstructed `SlotAction`. |
| `joint_ok(a,b)` | Two slot actions | `bool`; false only for globally impossible pairs. |
| `joint_index(a,b)` | Two slot actions | Flattened joint label `int` in `0..1520`. |
| `static_joint_mask()` | None | Cached NumPy `bool[39,39]`. |
| `_choice(a,slot,map_fn)` | Action, slot `0|1`, team-index-to-party-position callable | One Showdown command fragment `str`. |
| `to_choice_string(joint,map_fn)` | `JointAction`, mapping callable | Comma-separated Showdown choice `str`. |
| `legal_slot_actions(request,slot,map_fn)` | `Request`, slot, party-position-to-team-index callable | `list[SlotAction]`. |
| `legal_joint_actions(request,map_fn)` | `Request`, mapping callable | Position-legal `list[JointAction]`. |

### `config.py`

| function | input | output |
|---|---|---|
| `Config(...)` | Typed dataclass fields for paths/data/model/belief/search/self-play | Mutable configuration dataclass. |
| `_jsonable(v)` | Dataclass value, recursively | JSON-safe scalar/list/dict/string path. |
| `config_snapshot(cfg)` | `Config` | Typed JSON mapping `dict[str,Any]`. |
| `_parse_legacy_string(s)` | Legacy serialized `str` | Parsed bool/None/literal or original string. |
| `_coerce_like(default,value)` | Default typed value and decoded value | Value coerced to the default's type. |
| `config_from_snapshot(snapshot,base)` | Mapping (optionally under `config`) and optional base `Config` | Reconstructed `Config`, including conservative legacy defaults. |
| `load_config_snapshot(path,base)` | JSON path and optional base | `Config`. |
| `config_diff(a,b,fields)` | Two configs and optional field iterable | `list[(field, a_value, b_value)]`. |

### `env.py`

| function | input | output |
|---|---|---|
| `_write_atomic(path,content)` | Path-like, text `str` | `None`; atomically replaces file when content changed. |
| `spawn_node(cfg,js_name,js_source)` | `Config`, filename, JavaScript source | `(subprocess.Popen, stderr_tail_callable)`. |
| `Sidecar(cfg)` | Runtime config | RPC owner with one Node process. |
| `Sidecar.rpc(obj)` | JSON-safe request mapping | Decoded JSON response mapping; raises on process/RPC failure. |
| `Sidecar.close()` | None | `None`; closes owned Node process. |
| `SidecarBattle(sidecar,resp)` | Sidecar and create/restore response mapping | Battle wrapper with ids/log/requests/winner. |
| `SidecarBattle.create(sidecar,format_id,p1team,p2team)` | Sidecar, format id, two packed-team strings | New `SidecarBattle`. |
| `SidecarBattle.step(choices)` | `dict[SideID,str]` | Response mapping; mutates battle log/requests/end state. |
| `SidecarBattle.save()` | None | Opaque JSON-serializable simulator state. |
| `SidecarBattle.restore(sidecar,state)` | Sidecar and saved state | Independent `SidecarBattle` fork. |
| `SidecarBattle.destroy()` | None | `None`; frees simulator battle id. |
| `SidecarBattle.pending_sides()` | None | `list[SideID]` needing choices. |
| `pack_team(team)` | `list[PokemonSet]` | Showdown packed-team `str`. |
| `full_set(s)` | Partial sampled/full set mapping | Normalized `PokemonSet`. |
| `reconstruct(sidecar,format_id,tracker,teams,brought)` | Sidecar, format, `LogParser`, `dict[SideID,list[Set]]`, brought indices | `(SidecarBattle, dict[SideID,list[team_idx]])`. |
| `random_choice(request,rng)` | `Request`, random-like object | Legal-ish Showdown choice `str`. |
| `_step_random(battle,rng)` | Battle and RNG | `None`; advances once with default fallback. |
| `dump_dex(cfg)` | Config | `None`; writes `dex.json`. |
| `benchmark(cfg,n_steps,seed)` | Config and integer workload/seed | `None`; prints correctness/throughput. |
| `selftest(cfg)` | Config | `None`; asserts reconstruction behavior and prints result. |
| `make_live_player(sets,searcher,usage,cfg,on_decision,**kwargs)` | Own sets, `MoveChooser`, usage mapping, config, optional callback | Instance of a dynamically defined poke-env `Player` subclass. |
| `run_live(ckpt,team_packed,n_games,ladder,cfg)` | Checkpoint path, packed team, run options | `None`; runs async ladder/challenges. |

### `data.py`

| function | input | output |
|---|---|---|
| `sid(name)` / `base_species(species)` / `_forme_base(species)` | Display/species string | Lowercase id / explicit base species / base-form id. |
| `parse_packed_team(s)` | Showdown packed-team string | `list[PokemonSet]`. |
| `Mon(team_idx,set_)` | Team index and full set | Mutable public battle-mon tracker. |
| `Mon.view_own()` / `view_opp()` | None | `MonOwnView` / CTS-safe `MonOpponentView`. |
| `Side(team)` | `list[PokemonSet]` | Side tracker with mons/conditions/name lookup. |
| `Side.mon(nickname,details)` | Protocol nickname/details | Matching `Mon`, with active/first fallback. |
| `Side.active(slot)` | `0|1` | Active `Mon | None`. |
| `LogParser(tag,ts,log,fmt)` | Battle identity, Unix timestamp, raw log, format | Streaming/parser state. |
| `LogParser._reset_turn_track()` | None | `None`; resets action/order scratch fields. |
| `LogParser._pos(ref,details)` | Protocol ident and optional details | `(SideID, slot|None, Mon)`. |
| `LogParser._hp(s)` | Showdown condition string | `(hp_fraction:float, status:str)`. |
| `LogParser._reveal(...)` | Side, mon, reveal kind/name | `None`; updates mon and appends event. |
| `LogParser._reveal_from_tags(...)` | Protocol tags and default source | `None`; extracts `[from]/[of]` reveal. |
| `LogParser._spe_ctx(...)` | Side and mon | Speed context mapping `{spe,par,tw}`. |
| `LogParser._view(side)` | `SideID` | `PositionState`. |
| `LogParser._close_turn()` | None | `None`; finalizes actions/events for current turn. |
| `LogParser._open_turn(n)` | Turn integer | `None`; advances public counters and snapshots both views. |
| `LogParser.feed(line)` | One protocol line `str` | `bool` battle-decided flag. |
| `LogParser.drain_events()` | None | Pending `list[BeliefEvent]`, then clears it. |
| `LogParser.parse()` | Stored raw log | `ParsedBattle` mapping or `None` if incomplete. |
| `LogParser._event(cmd,parts,line)` | Parsed protocol command/list/raw line | `None`; mutates tracker/event/action state. |
| `download(cfg)` | Config | `None`; downloads configured log files. |
| `split_of(match_id,cfg)` | Match id and split config | `'train'|'val'|'test'`. |
| `iter_battles(*paths)` | Pickle paths | Iterator of parsed battle mappings. |
| `parse(cfg)` | Config | `None`; writes parsed pickles, vocab names, usage stats. |
| `battle_weight(rec,side,max_ts,cfg)` | Parsed battle, side, newest timestamp, config | Positive sample weight `float`. |
| `prep(cfg,resume)` | Config and resume flag | `None`; writes tokenized NPZ shards. Nested `flush` writes one shard; `n_transitions` returns an integer count. |
| `main` dispatch | CLI args | `None`; invokes download/parse/prep. |

### `damage.py`

| function | input | output |
|---|---|---|
| `DamageBridge(cfg)` | Config with Node paths | Owned calc subprocess and request cache. |
| `DamageBridge.calc_batch(reqs)` | `list[canonical request dict]` | `list[DamageCell|None]` in input order. |
| `DamageBridge.close()` | None | `None`; closes calc subprocess. |
| `request(attacker,defender,move,field,crit)` | Partial combatant mappings, move id, optional field, crit flag | Canonical JSON-safe calc request mapping. |
| `damage_features(state,belief,bridge)` | `PositionState`, `BeliefModel`, `DamageBridge` | `DamageFeatures`. |
| `_sid(name)` | Name string | Lowercase alphanumeric id. |

### `beliefs.py`

| function | input | output |
|---|---|---|
| `load_dex(cfg)` / `load_spreads(cfg)` | Config | Decoded asset mapping or `None` when optional/missing. |
| `_att_key`, `_arch_prior_mult` | Species/nature/dex or archetype/nature | Attack-stat id `str` / scalar prior multiplier. |
| `archetype_spread(arch,species,nature,dex)` | Archetype and species metadata | Six-int SP spread list. |
| `calc_stat(base,stat,nature,sp)` | Base stat, stat id, nature, SP | Champions integer stat. |
| `boost_mult(stage)` | Boost stage `-6..6` | Numeric multiplier `float`. |
| `OpponentBelief(opp_species,usage,cfg,bridge,my_team)` | Preview species ids, usage prior, config, optional calc bridge, known own sets | Mutable per-mon particle posterior. |
| `_free_sp`, `_spread_nature_combos`, `_expand_spreads`, `_expand_archetypes` | Particle/species/prior inputs | Bool, weighted combo list, or expanded particle lists. |
| `_base`, `_particle_speed`, `_my_speed` | Species/particle/context/SP values | Base stat or effective speed number/`None`. |
| `_apply`, `_apply_list` | Mon index, predicate/multiplier list, cause | `None`; normalize/update weights and diagnostics. |
| `_oracle_mass` | Mon index | Current oracle posterior mass `float`. |
| `_hard_ok` | Mon index and particle | `bool` against public hard reveals. |
| `_hard_depletion_fallback` | Mon index | Replacement normalized weight list. |
| `_resample_check` | Mon index | `None`; mixes prior when alive fraction is low. |
| `update(events,viewer)` | `list[BeliefEvent]`, viewer side | `None`; mutates all affected posteriors. |
| `_speed_evidence`, `_damage_evidence` | Decoded event/context/viewer data | `None`; applies evidence. |
| `_strict_speed_mults`, `_strict_attack_mults` | Mon/context/observation parameters | Per-particle survival `list[float]`. |
| `_atk_hypo`, `_as_attacker`, `_as_defender` | Mon/particle/context/SP | Canonical calc combatant mapping. |
| `_atk_slack`, `_bulk_slack` | Mon/particle/move | Conservative numeric slack multiplier. |
| `_species_cur` | Mon index | Current species id, including inferred mega. |
| `top_particle(k)` | Opponent team index | Highest-weight `ParticleSet`. |
| `summary()` | None | `BeliefSummary`. |
| `arch_posterior`, `nature_posterior`, `item_posterior` | Mon index | Descending `list[(label,probability)]`. |
| `sample_sets(n,rng)` | Count and random-like object | `list[list[PokemonSet]]` sampled in preview order. |
| `_sampled_evs(k,j,p)` | Mon/particle indices and particle | Six-int concrete spread or `None`. |
| `determinized(sets,cfg)` | Known full sets and config | Collapsed `OpponentBelief` with unit mass on truth. |
| `_feasible_sp_range(results,frac,tol,truncated)` | Ordered calc cells and observed damage parameters | Inclusive `(lo,hi)` SP indices or `None`. |
| `_quantile(sorted_pairs,q)` | `(value,weight)` pairs and quantile | Weighted quantile `float`. |
| `_sid`, `_set_key`, `_id_key` | Names/set mappings | Canonical id or hashable identity tuples. |
| `audit(max_battles,cfg)` | Optional cap and config | `None`; prints belief-quality metrics. |

### `tokenizer.py`

| function | input | output |
|---|---|---|
| `PositionTokenizer(vocab,lists,cfg,layout)` | Token-id mapping, namespace lists, config, layout `1|2|3` | Layout-aware tokenizer with offsets and lookup maps. |
| `build(cfg)` | Config plus `vocab_names.json` | New tokenizer; writes current-layout `vocab.json`. |
| `load(cfg,path)` | Config and optional explicit vocab path | Tokenizer reconstructed at the file's recorded layout. |
| `vocab_size()` | None | Vocabulary size `int`. |
| `decode(ids)` | Iterable/array of token ids | `list[str]` token names. |
| `move_idx`, `item_idx`, `ability_idx` | Namespace id string | Label index `int`, `0` for unknown. |
| `opp_species_positions()` | None | Six token positions `list[int]` used by aux heads. |
| `_t(name)` | Token name | Token id `int`, falling back to `UNK`. |
| `_slot_tok`, `_hp_tok`, `_boost_toks`, `_prot_tok` | Mon view mapping | Slot/HP/boost/Protect token name(s). |
| `_dmg_toks(cell)` / `_dmg_bound_tok(v)` | `DamageCell|None` / fraction | Damage token names / one token name. |
| `encode(state,belief_summary,dmg)` | `PositionState`, `BeliefSummary`, `DamageFeatures` | NumPy `uint16[n_tokens]`; asserts exact layout length. |
| `active_dmg_grid(state,dmg)` | State and damage feature mapping | NumPy `uint8[2,4,2]` mean active damage percent. |
| `sid_of(name)` | Display name | Lowercase alphanumeric id. |

### `models/policy_value.py`

| function | input | output |
|---|---|---|
| `PolicyValueNet(...)` | Vocabulary/layout/aux sizes, config, `slot|joint` head, optional model config | Torch `nn.Module` with stored hyperparameters. |
| `forward(tokens)` | Torch `long[B,T]` | `(policy_logits, value[B], (item_logits[B,6,I], ability_logits[B,6,A], move_logits[B,6,M]))`; policy is `[B,1521]` joint or legacy `[B,2,39]`. |
| `joint_dist(pol)` | Either policy-logit shape | Torch normalized `[B,1521]`. |
| `predict_batch(tokens)` | NumPy/Torch integer `[B,T]` | `ModelPrediction` NumPy tuple. |
| `save(path)` | Destination path | `None`; writes hp/state/config checkpoint. |
| `load(path,cfg,device)` | Checkpoint path, base config, Torch device | Loaded/eval-capable `PolicyValueNet`. |
| `from_slot(slot_model,cfg)` | Legacy slot-head model | Joint-head model initialized to the same distribution. |
| `clean_state_dict(model)` | Eager or compiled module | Eager-keyed state mapping. |
| `strip_compile_prefix(state)` | State mapping | Copy with `_orig_mod.` prefixes removed. |
| `_model_cfg_from_checkpoint(hp,state,cfg)` | Saved metadata/state/base config | Recovered model-architecture mapping. |

### `models/baselines.py`

| function | input | output |
|---|---|---|
| `RandomPolicy.predict_batch(dmg_active)` | NumPy batch, first dimension `B` | Uniform factorized NumPy float `[B,2,39]`. |
| `MaxDamagePolicy(eps)` | Smoothing probability | Stateless predictor. |
| `MaxDamagePolicy.predict_batch(dmg_active)` | NumPy `uint8[B,2,4,2]` | Smoothed factorized NumPy float `[B,2,39]`. |

### `train.py`

| function | input | output |
|---|---|---|
| `Shards(split,cfg)` | Split name and config | In-memory concatenated BC dataset arrays. |
| `Shards.__len__()` | None | Transition count `int`. |
| `Shards.__getitem__(idxs)` | Scalar/list/NumPy batch indices | Seven Torch tensors in the BC batch contract. |
| `make_loader(ds,batch_size,shuffle,device,cfg)` | Batch-index dataset and loader options | Torch `DataLoader` yielding already-batched tuples. |
| `compute_loss(model,batch,cfg)` | Policy/value model and BC batch | `(scalar loss tensor, dict[str, detached scalar tensor])`. |
| `run_epoch(model,loader,device,opt,sched,cfg)` | Model, loader, device, optional optimizer/scheduler | Mean metric mapping `dict[str,float]`; trains iff `opt` supplied. |
| `main(cfg)` | Config plus optional CLI epoch count | `None`; builds/resumes model, trains, validates, checkpoints. |

### `evaluate.py`

| function | input | output |
|---|---|---|
| `factorized_to_joint(slot_dists)` | NumPy `[B,2,39]` | Masked normalized NumPy `[B,1521]`. |
| `score(flat,acts)` | Joint probabilities `[B,1521]`, slot labels `[B,2]` | Metric mapping (top-k/perplexity/ECE/recall fields). |
| `_blocks(tok,names,base)` | Tokenizer, field names, block offset | Human-readable decoded field strings. |
| `_short(tok_name)` | Namespaced token `str` | Short display string. |
| `describe_state(tok,ids)` | Tokenizer and one token vector | Multiline position description `str`. |
| `describe_action(tok,ids,pair)` | Tokenizer, tokens, two slot labels | Human-readable joint action `str`. |
| `worst(flat,acts,ds,tok,n)` | Predictions, labels, dataset, tokenizer, count | `None`; prints lowest-probability examples. |
| `switch_report(net,acts,k)` | Joint predictions, labels, k | `None`; prints switch rate/recall. |
| `aux_report(model,ds,cfg)` | Model, test shards, config | `None`; prints item/ability/move aux metrics. |
| `main(cfg)` | Config and optional positional checkpoint/flags | `None`; loads test shards/model and prints reports. |

### `agents/interfaces.py`

| method | input | output |
|---|---|---|
| `MoveChooser.choose(...)` | Tracker, external `BeliefModel`, side, raw request, brought indices, optional opponent indices/temperature/root noise | `(JointAction, ChoiceInfo)`. |
| `BeliefModel.update(events,viewer)` | New `BeliefEvent` sequence and viewer side | `None`; posterior mutation. |
| `BeliefModel.summary()` | None | `BeliefSummary`. |
| `BeliefModel.sample_sets(n,rng)` | Count/RNG | `list[list[PokemonSet]]`. |
| `BeliefModel.top_particle(k)` | Team index | Particle mapping. |
| `PositionEncoder.encode(...)` | Tracker/side/belief/optional cached summary | NumPy token vector. |
| `PolicyPrior.legal_priors(...)` | Flat model distribution, legal actions, optional k | `(probability_array, retained_action_list)`. |
| `LeafEvaluator.predict_batch(tokens)` | `[B,T]` tokens | `ModelPrediction`. |
| `LeafEvaluator.value(values,index)` | Value array/index | Python `float`. |
| `LeafEvaluator.terminal_value(winner,my,opp)` | Winner/side ids | `+1.0|-1.0|0.0`. |
| `Searcher.run(...)` | Chooser hook host, `DetGame` list, per-det budget | `None`; mutates trees. |
| `Searcher.aggregate_root(...)` | Determinizations and policy-only flag | Sorted root aggregate rows. |

### Versioned agent implementations

| function | input | output |
|---|---|---|
| `DeterminizedDUCTChooser(...)` | Same constructor inputs as legacy `search.mcts.Searcher` | Full versioned v1 chooser; external belief class pinned to v1. |
| `PolicyOnlyChooser(chooser)` | Full chooser | Wrapper sharing bridge/belief/resource ownership. |
| `PolicyOnlyChooser.choose(...)` | Standard chooser inputs | Wrapped `(JointAction, ChoiceInfo)` using root priors only. |
| `PolicyOnlyChooser.close()` | None | `None`; closes wrapped chooser. |
| `RandomChooser(rng)` | Optional random-like object | Resource-free random chooser. |
| `RandomChooser.choose(...)` | Standard inputs; belief/search knobs ignored | Uniform legal `(JointAction, ChoiceInfo)`. |
| `RandomChooser.close()` | None | `None`. |
| `MaxDamageChooser(cfg)` | Config | Chooser owning `DamageBridge`. |
| `MaxDamageChooser.choose(...)` | Standard inputs | Greedy per-slot damage `(JointAction, ChoiceInfo)`. |
| `MaxDamageChooser.close()` | None | `None`; closes bridge. |
| `single_action_info(description,value)` | Text and scalar | Baseline `ChoiceInfo` mapping. |

### Versioned bricks

| function | input | output |
|---|---|---|
| `TokenPositionEncoder(tokenizer,damage_bridge)` | Tokenizer and optional non-owned bridge | Encoder brick. |
| `position(tracker,side,belief)` | Tracker/side/belief | `(PositionState, DamageFeatures)`. |
| `encode_position(position,summary)` | Prior tuple and belief summary | NumPy token vector. |
| `encode(...)` | Full encoder protocol inputs | NumPy token vector. |
| `PolicyValuePrior.legal_priors(...)` | Flat `[1521]`, legal joints, optional k | Normalized `float64[K]`, retained actions. |
| `PolicyValueLeafEvaluator(model)` | Model/BatchedEvaluator or `None` in terminal-only solve | Evaluator brick. |
| `predict_batch(tokens)` | `[B,T]` | Model prediction tuple; raises when model absent. |
| `value(values,index)` | Value batch/index | Python scalar. |
| `terminal_value(winner,my,opp)` | Side ids | Exact oriented scalar. |
| `DecoupledUCTSearcher.run(...)` | Chooser, determinizations, budget | `None`; repeated simulation. |
| `aggregate_root(...)` | Determinizations/policy flag | Sorted root rows. |
| `simulate(chooser,det)` | Hook host and one `DetGame` | `None`; mutates tree/health. |
| `leaf_value(chooser,det,battle,tracker,leaf)` | Current rollout objects | Oriented `float` terminal or neural leaf value. |
| `settle(chooser,battle,tracker)` | Current rollout objects | `None`; plays forced switches in place. |

### `agents/spec.py` and `agents/registry.py`

| function | input | output |
|---|---|---|
| `BrickSpec.from_dict(value)` | Existing spec or decoded mapping | `BrickSpec`; raises on missing impl. |
| `BrickSpec.to_dict()` | None | Compact JSON-safe mapping. |
| `AgentSpec.__post_init__()` | Constructed fields | `None`; validates schema/identity. |
| `AgentSpec.from_dict(value)` | Existing spec or decoded mapping | `AgentSpec`. |
| `AgentSpec.load(path)` | JSON path | `AgentSpec`. |
| `AgentSpec.to_dict()` | None | Complete `AgentSpecJson`. |
| `AgentSpec.dump(path)` | Destination path | `None`; writes JSON. |
| `AgentSpec.resolve(bundle_dir,relative_path)` | Bundle root and manifest path | Safe resolved `Path|None`; rejects traversal. |
| `AgentSpec.behavior_paths()` | None | `set[str]` of every referenced behavior file. |
| `default_duct_spec(cfg,runtime,source,archive)` | Config and optional metadata mappings | Complete DUCT-v1 `AgentSpec`. |
| `config_from_agent_spec(cfg,spec)` | Runtime config and spec | Config with authoritative brick values; rejects conflicts/unknowns. |
| `AgentRegistry()` | None | Empty explicit agent/brick maps. |
| `register_agent`, `register_brick` | Implementation id and class | `None`; mutate registry. |
| `validate(spec)` | Spec/mapping | Parsed `AgentSpec`; raises on unavailable or missing IDs. |
| `build(spec,model,tokenizer,cfg,seed,debug,sidecar,apply_spec_config)` | Manifest plus runtime dependencies | Concrete `MoveChooser`. |
| `implementation_source_hashes(spec,repo_root)` | Spec and optional source root | `dict[relative_path, sha256_hex]`. |
| `verify_implementation_sources(spec,repo_root)` | Spec/root | `None` or raises on mismatch. |
| `build_agent(spec,**kwargs)` | Same as registry build | Concrete chooser. |

### `agents/evaluation.py`

| function | input | output |
|---|---|---|
| `BrickEvaluation(...)` | Implementation/suite/metrics/case count plus metadata | JSON-serializable result dataclass. |
| `EvaluationStore(path,cfg)` | Optional JSONL path/config | Append-only store. |
| `append(result)` | `BrickEvaluation` | Same result after one JSONL append. |
| `load(brick_impl,suite)` | Optional filters | `list[decoded result dict]`. |
| `_save(...)` | Result fields/store/config | Appended `BrickEvaluation`. |
| `evaluate_policy_prior(...)` | Prior, `(distribution,legal,target)` cases, k | Saved result with recall/normalization/NLL/ECE/latency. |
| `evaluate_position_encoder(...)` | Encoder and `(tracker,side,belief,expected_tokens)` cases | Saved exactness/latency result. |
| `evaluate_leaf_evaluator(...)` | Evaluator and `(tokens,target_values)` batches | Saved MAE/terminal-sign/latency result. |
| `evaluate_belief_model(...)` | Belief and `(events,viewer,index,oracle_subset)` cases | Saved oracle/depletion/latency result. |
| `evaluate_searcher(...)` | Searcher and `(chooser,dets,budget,predicate)` cases | Saved pass-rate/sims-per-second/latency result. |

### `search/node.py`, `search/mcts.py`, and `search/debug.py`

| function | input | output |
|---|---|---|
| `Node(my_actions,opp_actions,my_priors,opp_priors,value)` | Two joint-action lists, parallel probability arrays, optional leaf scalar | Mutable decoupled-UCT node. |
| `Node._pick(p,n,w,total,c_puct)` | Parallel prior/count/value arrays and exploration values | Selected action index `int`. |
| `Node.select(c_puct)` | Exploration scalar | `(my_action_index, opp_action_index)`. |
| `Node.update(i,j,z)` | Selected indices and searching-side result | `None`; increments both zero-sum bandit tables. |
| `_pos_maps(request,name_to_idx)` | Raw request and set-name-to-preview-index mapping | `(party_pos->team_idx callable, dict[team_idx,party_pos])`. |
| `joint_choice(request,joint,name_to_idx)` | Request, joint action, identity mapping | Showdown choice string. |
| `_joint_priors(joint_dist,joints,k)` | Flat model distribution, legal actions, optional k | Compatibility delegation to v1 prior. |
| `DetGame(searcher,tracker,opp_sample,my_id,my_request,my_brought,opp_brought,solve)` | Chooser orchestration inputs and one sampled team | Reconstructed determinization object/root node. |
| `Searcher(model,tok,cfg,seed,debug,sidecar,position_encoder,policy_prior,leaf_evaluator,searcher)` | Inference/tokenizer/runtime and optional injected bricks | Legacy-compatible full chooser; owns sidecar/bridge only when not injected. |
| `Searcher.close()` | None | `None`; closes owned processes. |
| `Searcher.choose(...)` | Standard chooser inputs plus legacy `policy_only` flag | `(JointAction, ChoiceInfo)`. |
| `_debug_print(dets,belief,wall,policy_only)` | Completed roots and timing | `None`; prints profiler/root/belief diagnostics. |
| `_simulate`, `_leaf_value`, `_settle` | Compatibility arguments | Delegated `None`/`float` results from search brick. |
| `_expand(det,battle,tracker,my_request)` | Determinization and current sim/tracker state | New `Node` with encoded/evaluated/pruned actions. |
| `_describe(tracker,side,joint)` | Public tracker, side, joint action | Human-readable action string. |
| `SearchDebug(enabled)` | Bool | Timer/counter object. |
| `SearchDebug.__call__(name)` | Phase name | Context manager (real timer or null context). |
| `SearchDebug._timer(name)` | Phase name | Yielding context manager; records elapsed/call count. |
| `SearchDebug.report(wall)` | Total seconds | Multiline phase table `str`. |
| `maybe_cprofile(path)` | Optional output path | Context manager; optionally writes pstats. |
| `belief_data(belief,oracle)` | Belief and optional true sets | `list[per-mon diagnostic dict]`. |
| `belief_report(...)` | Same | Multiline human report `str`. |
| `root_table(dets,describe,top)` | Determinizations, description callback, row cap | Multiline root-stat table `str`. |

### `benchmark.py`

| function | input | output |
|---|---|---|
| `era_hash(cfg)` | Config | Ten-hex SHA-1 behavior-era label. |
| `git_commit()` / `git_dirty()` | None | Short commit `str` / dirty `bool`. |
| `runtime_identities(cfg)` | Config/current installation | Nested JSON mapping for format, engines, Python/Torch/NumPy. |
| `bench_dir`, `registry_path` | Config | Corresponding `Path`. |
| `load_registry(cfg)` | Config | Decoded `{'results': list[BenchmarkResult]}` or empty default. |
| `save_registry(reg,cfg)` | Registry mapping/config | `None`; writes JSON. |
| `archive(name,ckpt,notes,cfg)` | Unique name, optional explicit checkpoint, notes, config | `None`; creates immutable full-agent directory or raises. |
| `rename(old,new,cfg)` | Archive names/config | `None`; renames bundle and migrates result references. |
| `list_bundles(cfg)` | Config | `None`; prints directories containing metadata. |
| `Contestant(name,cfg,device)` | `'current'` or archive name, config, Torch device | Loaded model/tokenizer/spec/config/usage holder. |
| `_verify_runtime(spec,current_cfg)` | Static manifest/current environment | `None`; raises on recorded mismatch. |
| `_runtime_cfg(saved,current,bundle,strict)` | Saved/current configs, optional archive root | Config with frozen assets and machine-local paths. |
| `_load_usage(search_cfg,current_cfg,strict)` | Configs/strict flag | Decoded usage mapping. |
| `_with_runtime_overrides(saved,current,sims,depth)` | Configs and explicit run overrides | Replaced runtime `Config`. |
| `_print_cfg_diffs(a,b,current)` | Two contestants/current config | `None`; prints meaningful differences. |
| `_make_searcher(contestant,run_cfg)` | Loaded contestant and run config | Exact registered `MoveChooser` or legacy searcher. |
| `run_game(sc,bots,sets_by_side,cfg,temperature,rng,max_turns,feed)` | Shared sidecar, side->Bot, true sets, options, optional spectator feed | `(winner_side|None, turns:int)`. |
| `series_pairings(team_names,repeat,quick,seed)` | Names/count/subsample options | Ordered `list[(team_a,team_b)]`. |
| `run_series(...)` | Contestant names, budgets/workers/replay/record filters | `list[BenchmarkResult]`; optionally appends registry/replays. |
| `report(name_a,name_b,results)` | Labels and game rows | `None`; prints score/CI/Elo/team split. |
| `wilson(w,n,z)` | Score successes, count, z | `(lower,upper)` confidence bounds. |
| `elo_diff(score)` | Score fraction | Elo difference `float` with endpoint clipping. |
| `standings(cfg)` | Config | `None`; prints era-separated, architecture-grouped BT ratings. |
| `main(cfg)` | Config and CLI args | `None`; dispatches archive/list/rename/play/standings. |

### `selfplay.py`

| function | input | output |
|---|---|---|
| `sp_dir(cfg)` / `buffer_dir(cfg)` | Config | Self-play checkpoint/buffer `Path`. |
| `BatchedEvaluator(model,max_batch,wait_ms)` | Predict-batch model and batching thresholds | Thread-safe inference queue wrapper. |
| `predict_batch(tokens)` | NumPy-like `[B,T]` | Same `ModelPrediction` contract; blocks until worker responds. |
| `_loop()` | Queue state | Infinite daemon loop; batches requests and fulfills per-call dict/events. |
| `RecorderBot(...,noise)` | `observe_game.Bot` inputs plus root-noise tuple | Bot with mutable `list[SelfPlaySample]`. |
| `RecorderBot.decide(request,temperature)` | Raw request and scalar temperature | `(Showdown choice str, ChoiceInfo)` and appends one sample. |
| `play_selfplay_game(sc,bots,sets_by_side,cfg,rng,max_turns,feed)` | Game objects/options | Winner `SideID|None`; records through bots/feed. |
| `oracle_labels(tok,opp_sets)` | Tokenizer and true opponent sets | `(item_indices, ability_indices, move_index_rows)`. |
| `pad6(labels,fill)` | List and fill value | Exactly six entries. |
| `generate_games(model,tok,cfg,n_games,workers,seed,verbose)` | Model/tokenizer/generation options | `SelfPlayShard` of stacked NumPy arrays ready for NPZ. Nested `flush` converts visits/oracle labels; worker returns via shared output. |
| `_gen_subprocess(cfg)` | Config and CLI process arguments | `None`; loads model/generates/writes one subprocess shard. |
| `SelfPlayShards(cur_iter,cfg)` | Current iteration/config | Dataset concatenating configured replay-buffer window. |
| `__len__()` / `__getitem__(idxs)` | None / batch indices | Sample count / seven-tensor self-play batch. |
| `sp_loss(model,batch,cfg)` | Joint model and self-play batch | `(loss tensor, detached metric tensor mapping)`. |
| `train_iteration(model,cur_iter,device,cfg)` | Model/iteration/device/config | `None`; fine-tunes model in place and prints metrics. |
| `gate(model_new,model_old,tok,cfg,n_games,workers)` | Two models and gate options | New-model score fraction `float`. |
| `fork_model(src_ckpt,cfg,device)` | Source checkpoint/config/device | Loaded joint-head `PolicyValueNet`, converting legacy head if needed. |
| `main(cfg)` | Config and CLI flags | `None`; controls resumable generate/train/gate iterations. |

### `observe_game.py`

| function | input | output |
|---|---|---|
| `cts_placeholder(set)` | True full set | Redacted preview-safe `PokemonSet`. |
| `Bot(side,my_sets,opp_sets,searcher,usage,cfg,debug)` | Side id, true own/preview opponent sets, chooser/assets/config | Tracker + external belief game bot. |
| `Bot.feed(lines)` | Protocol line iterable | `None`; mutates tracker. |
| `Bot.decide(request,temperature)` | Raw request/scalar | `(Showdown choice str, ChoiceInfo)`. |
| `Bot.show(info)` | Choice diagnostics | `None`; prints state/belief/strategy. |
| `play_game(sc,bots,teams,cfg,step_mode,temperature,p2_random,rng)` | Sidecar/bots/true teams/options | Winner `SideID|None`. |
| `main(cfg)` | Config/CLI flags | `None`; loads chooser and runs observed game. |

### `play.py`

| function | input | output |
|---|---|---|
| `_sprite(species_name)` | Display species | Sprite slug `str`. |
| `on_decision(battle,g,info)` | poke-env battle, live game mapping, `ChoiceInfo` | `None`; replaces dashboard state. |
| `start_dashboard(port)` | TCP port | Started `ThreadingHTTPServer`. Nested handler returns HTTP responses. |
| `port_open(port)` | TCP port | `bool`. |
| `start_showdown(cfg)` | Config | `subprocess.Popen|None`; reuses already-open port. |
| `pick(prompt,options)` | Prompt and `(value,description)` options | Selected value. |
| `build_chooser(kind,ckpt,cfg,debug)` | Chooser label/checkpoint/config/debug | `MoveChooser`. |
| `main(cfg)` | Config/CLI flags | `None`; orchestrates team/client/server/live games/dashboard. |

### `scenarios.py`

| function | input | output |
|---|---|---|
| `mon(species,moves,item,ability,nature,evs,gender,level)` | Authored set fields | Normalized `PokemonSet`. |
| `filler()` | None | Pre-fainted scenario filler set. |
| `_move_marginals(info,moves)` | `ChoiceInfo`, move substrings | Probability mapping by substring. |
| `_mass(info,sub)` | Choice info and substring | Matching strategy mass `float`. |
| `_slot_mass(info,slot,prefix)` | Choice info, slot index, prefix(es) | Slot marginal mass `float`. |
| `_joint_mass(info,slot_a,prefixes_a,slot_b,prefixes_b)` | Choice info and two slot predicates | Joint strategy mass `float`. |
| `build_tracker(p1_sets,p2_sets,hp,fainted,cfg,weather)` | Scenario teams/public overrides/config | Seeded `LogParser`. |
| `print_damage_matrix(searcher,p1_sets,p2_sets)` | Chooser and true sets | `None`; prints pairwise cells. |
| `run_scenarios(searcher,cfg)` | Chooser/config | Failure count `int`. |
| `mine(cfg,max_out)` | Config/count | `None`; writes candidate `endgames.json`. |
| `_apply_view(side,views)` | Tracker `Side`, stored mon-view mappings | `None`; restores public attributes. |
| `_infer_brought(mons)` | Mon sequence | Team-index `list[int]` inferred from appeared/nonfainted mons. |
| `replay(searcher,cfg,i)` | Chooser/config/candidate index | `None`; prints one replay decision. |
| `main()` | CLI flags | `None` or process exit code via `sys.exit` after assertions. |

### `spectate.py`

| function | input | output |
|---|---|---|
| `resolve_public(lines)` | Raw split-protocol line list | Public-only `list[str]`, resolving `|split|` pairs. |
| `_slug(s)` | Arbitrary label | Filesystem-safe slug `str`. |
| `_rename_players(lines,side_names)` | Protocol lines and `SideID->name` | Renamed line list. |
| `write_replay(path_stem,pub_lines,header,rid)` | Output stem, public log, metadata, replay id | `None`; writes `.log` and standalone `.html`. |
| `GameFeed(spectator,gid,meta)` | Parent spectator, game id, metadata | Thread-safe per-game feed. |
| `feed(raw_lines)` | New raw lines | `None`; appends public lines and publishes snapshot. |
| `finish(winner_side)` | Side id/None | `None`; marks done and optionally writes replay. |
| `public_lines()` | None | Copy `list[str]`. |
| `Spectator(run_name,cfg,live,port,save)` | Run label/config/live/save options | Multi-game dashboard/replay coordinator. |
| `new_game(a,b,team_a,team_b,side_of,fmt)` | Contestant/team/side/format metadata | New `GameFeed`. |
| `_start(port)` | TCP port | `None`; starts daemon HTTP server. Nested handler emits index/game JSON. |

### `teams.py`

| function | input | output |
|---|---|---|
| `parse_export(text)` | Showdown export text | `list[PokemonSet]`. |
| `export_text(sets)` | Set sequence | Showdown export `str`. |
| `get(name)` | Team registry key | Deep-copied `list[PokemonSet]`. |
| `menu()` | None | `list[(team_name, description)]`. |
| `validate(cfg)` | Config | Does not return normally; submits every team, prints results, then `sys.exit(0|1)`. |
| `mine(n,cfg)` | Result count/config | `None`; prints common parsed teams. |

### `build_spreads.py`

| function | input | output |
|---|---|---|
| `_sid(name)` | Name | Lowercase alphanumeric id. |
| `_get(url,timeout)` | URL/seconds | UTF-8 response text `str`; network errors propagate. |
| `discover_key(fmt)` | Format id | Pikalytics dataset key `str`. |
| `candidate_dates()` | Current date | Descending candidate month strings. |
| `fetch_format(fmt)` | Format id | `(date:str, discovered_key:str, list[per-mon mapping])`. |
| `_parse_ev(s)` | Slash-separated spread string | Valid six-int SP list or `None`. |
| `build(cfg)` | Config | `None`; fetches/writes normalized `spreads.json`. |

## CLI-only and test functions

CLI `main` blocks in `env.py`, `beliefs.py`, `benchmark.py`, `build_spreads.py`,
`evaluate.py`, `observe_game.py`, `play.py`, `scenarios.py`, `selfplay.py`,
`teams.py`, and `train.py` read `sys.argv`, print/write their documented
artifacts, and return `None` unless they terminate through `sys.exit`.

Test functions take no values except pytest-managed `tmp_path`, `sc`, or
`bridge` fixtures and return `None`; success is no exception. `sc` yields a
module-scoped `Sidecar`, `bridge` a module-scoped `DamageBridge`, and both are
closed by `tests/conftest.py`. The tiny archive fixture contains a schema-v1
random chooser and empty config; it returns the only legal pass/pass action in
its deterministic scenario.
