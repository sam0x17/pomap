//! Per-size graph benchmarks for all operation types.
//!
//! Usage: cargo bench --bench bench_graph -- <benchmark>
//!
//! Benchmarks: insert_allocate, insert_preallocated, get_hits, get_misses,
//!             update_existing, get_hotset, remove_hits, remove_misses, shrink_to

use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
    hint::black_box,
    io::Write as _,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

#[cfg(feature = "tui")]
use std::{
    io,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

use ahash::AHasher;
#[cfg(feature = "tui")]
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use hashbrown::HashMap as HashbrownMap;
use pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
#[cfg(feature = "tui")]
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout as RataLayout},
    style::{Color, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, Paragraph},
};

// ---------------------------------------------------------------------------
// Tracking allocator
// ---------------------------------------------------------------------------

static NET_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

struct TrackingAlloc;

unsafe impl GlobalAlloc for TrackingAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            NET_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        NET_ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            NET_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size > layout.size() {
                NET_ALLOCATED.fetch_add(new_size - layout.size(), Ordering::Relaxed);
            } else {
                NET_ALLOCATED.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL_ALLOC: TrackingAlloc = TrackingAlloc;

// ---------------------------------------------------------------------------
// Types & helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "bench-string")]
type BenchType = String;
#[cfg(not(feature = "bench-string"))]
type BenchType = u64;

type BenchKey = BenchType;
type BenchValue = BenchType;
type BenchHasherBuilder = BuildHasherDefault<AHasher>;
type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchPoMap3 = pomap::pomap3::PoMap3<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchHashMap = HashMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchHashbrownMap = HashbrownMap<BenchKey, BenchValue, BenchHasherBuilder>;

fn random_bench_item(rng: &mut StdRng) -> BenchType {
    #[cfg(feature = "bench-string")]
    {
        use rand::distr::Alphanumeric;
        (0..128).map(|_| rng.sample(Alphanumeric) as char).collect()
    }
    #[cfg(not(feature = "bench-string"))]
    {
        rng.random()
    }
}

fn random_items(seed: u64, count: usize) -> Vec<BenchType> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| random_bench_item(&mut rng)).collect()
}

fn generate_miss_keys(
    seed: u64,
    count: usize,
    present: &HashSet<BenchKey, BenchHasherBuilder>,
) -> Vec<BenchKey> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let k = random_bench_item(&mut rng);
        if !present.contains(&k) {
            out.push(k);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Constructor shorthands
// ---------------------------------------------------------------------------

macro_rules! pm  { ()          => { BenchPoMap::with_hasher(BenchHasherBuilder::default()) };
                   ($cap:expr) => { BenchPoMap::with_capacity_and_hasher($cap, BenchHasherBuilder::default()) }; }
macro_rules! pm3 { ()          => { BenchPoMap3::with_hasher(BenchHasherBuilder::default()) };
                   ($cap:expr) => { BenchPoMap3::with_capacity_and_hasher($cap, BenchHasherBuilder::default()) }; }
macro_rules! std { ()          => { BenchHashMap::with_hasher(BenchHasherBuilder::default()) };
                   ($cap:expr) => { BenchHashMap::with_capacity_and_hasher($cap, BenchHasherBuilder::default()) }; }
macro_rules! hb  { ()          => { BenchHashbrownMap::with_hasher(BenchHasherBuilder::default()) };
                   ($cap:expr) => { BenchHashbrownMap::with_capacity_and_hasher($cap, BenchHasherBuilder::default()) }; }

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const MAX_SIZE: usize = 10_000_000;
const NUM_POINTS_PER_DECADE: usize = 60;
const ROUNDS: u64 = 2;
/// Total operations per (size, impl) measurement for get/update/remove benchmarks.
const OPS_TARGET: usize = 1_000_000;
const HOT_SET_SIZE: usize = 1_000;
const SHRINK_OVER_ALLOC: usize = 8;

const IMPL_NAMES: [&str; 4] = ["pomap", "std_hashmap", "hashbrown", "pomap3"];

fn target_sizes() -> Vec<usize> {
    let mut power_targets = Vec::new();
    let mut current = 10usize;
    while current < MAX_SIZE {
        power_targets.push(current);
        current *= 10;
    }
    power_targets.push(MAX_SIZE);
    let rounds = NUM_POINTS_PER_DECADE.max(2);
    let mut targets = Vec::new();
    for window in power_targets.windows(2) {
        let (start, end) = (window[0], window[1]);
        for step in 0..rounds {
            let size = start + (end - start) * step / (rounds - 1);
            if targets.last().copied() != Some(size) {
                targets.push(size);
            }
        }
    }
    if targets.is_empty() {
        targets.push(MAX_SIZE);
    }
    targets
}

// ---------------------------------------------------------------------------
// Per-operation timing macros
// ---------------------------------------------------------------------------

/// Time cold inserts (no pre-allocation). Returns ns/insert.
macro_rules! time_insert_alloc {
    ($size:expr, $keys:expr, $values:expr, $ctor:expr) => {{
        let size = $size;
        let reps = (MAX_SIZE / size).max(1);
        let total_inserts = reps as u128 * size as u128 * ROUNDS as u128;
        let mut total_ns = 0u128;
        for _ in 0..ROUNDS {
            let start = Instant::now();
            for _ in 0..reps {
                let mut map = $ctor;
                for i in 0..size {
                    black_box(map.insert($keys[i].clone(), $values[i].clone()));
                }
                black_box(&map);
            }
            total_ns += start.elapsed().as_nanos();
        }
        total_ns as f64 / total_inserts as f64
    }};
}

/// Time pre-allocated inserts. Returns ns/insert.
macro_rules! time_insert_prealloc {
    ($size:expr, $keys:expr, $values:expr, $ctor:expr) => {{
        let size = $size;
        let reps = (MAX_SIZE / size).max(1);
        let total_inserts = reps as u128 * size as u128 * ROUNDS as u128;
        let mut total_ns = 0u128;
        for _ in 0..ROUNDS {
            let start = Instant::now();
            for _ in 0..reps {
                let mut map = $ctor;
                for i in 0..size {
                    black_box(map.insert($keys[i].clone(), $values[i].clone()));
                }
                black_box(&map);
            }
            total_ns += start.elapsed().as_nanos();
        }
        total_ns as f64 / total_inserts as f64
    }};
}

/// Time random gets on a pre-built map. Returns ns/get.
macro_rules! time_gets {
    ($size:expr, $keys:expr, $values:expr, $lookup_keys:expr, $lookup_range:expr, $seed:expr, $ctor:expr) => {{
        let size = $size;
        let mut map = $ctor;
        for i in 0..size { map.insert($keys[i].clone(), $values[i].clone()); }
        let total_ops = OPS_TARGET as u128 * ROUNDS as u128;
        let mut total_ns = 0u128;
        for _ in 0..ROUNDS {
            let mut rng = StdRng::seed_from_u64($seed ^ size as u64);
            let start = Instant::now();
            for _ in 0..OPS_TARGET {
                let idx = rng.random_range(0..$lookup_range);
                black_box(map.get(&$lookup_keys[idx]));
            }
            total_ns += start.elapsed().as_nanos();
        }
        total_ns as f64 / total_ops as f64
    }};
}

/// Time random updates via get_mut. Returns ns/update.
macro_rules! time_updates {
    ($size:expr, $keys:expr, $values:expr, $update_values:expr, $ctor:expr) => {{
        let size = $size;
        let mut map = $ctor;
        for i in 0..size { map.insert($keys[i].clone(), $values[i].clone()); }
        let total_ops = OPS_TARGET as u128 * ROUNDS as u128;
        let mut total_ns = 0u128;
        for _ in 0..ROUNDS {
            let start = Instant::now();
            for i in 0..OPS_TARGET {
                let key = &$keys[i % size];
                if let Some(v) = map.get_mut(key) {
                    *v = $update_values[i % size].clone();
                    black_box(&*v);
                }
            }
            total_ns += start.elapsed().as_nanos();
        }
        total_ns as f64 / total_ops as f64
    }};
}

/// Time removes of existing keys (clones map per batch). Returns ns/remove.
macro_rules! time_remove_hits {
    ($size:expr, $keys:expr, $values:expr, $ctor:expr) => {{
        let size = $size;
        let mut base = $ctor;
        for i in 0..size { base.insert($keys[i].clone(), $values[i].clone()); }
        let removes_per_clone = OPS_TARGET.min(size);
        let num_clones = (OPS_TARGET / removes_per_clone).max(1) * ROUNDS as usize;
        let total_removes = num_clones as u128 * removes_per_clone as u128;
        let mut total_ns = 0u128;
        for _ in 0..num_clones {
            let mut map = base.clone();
            let start = Instant::now();
            for i in 0..removes_per_clone {
                black_box(map.remove(&$keys[i]));
            }
            total_ns += start.elapsed().as_nanos();
        }
        total_ns as f64 / total_removes as f64
    }};
}

/// Time remove attempts with miss keys (non-destructive). Returns ns/remove.
macro_rules! time_remove_misses {
    ($size:expr, $keys:expr, $values:expr, $miss_keys:expr, $seed:expr, $ctor:expr) => {{
        let size = $size;
        let mut map = $ctor;
        for i in 0..size { map.insert($keys[i].clone(), $values[i].clone()); }
        let total_ops = OPS_TARGET as u128 * ROUNDS as u128;
        let mut total_ns = 0u128;
        for _ in 0..ROUNDS {
            let mut rng = StdRng::seed_from_u64($seed ^ size as u64);
            let start = Instant::now();
            for _ in 0..OPS_TARGET {
                let idx = rng.random_range(0..size);
                black_box(map.remove(&$miss_keys[idx]));
            }
            total_ns += start.elapsed().as_nanos();
        }
        total_ns as f64 / total_ops as f64
    }};
}

/// Time shrink_to on over-allocated maps. Returns ns/shrink.
macro_rules! time_shrink {
    ($size:expr, $keys:expr, $values:expr, $ctor:expr) => {{
        let size = $size;
        let mut base = $ctor;
        for i in 0..size { base.insert($keys[i].clone(), $values[i].clone()); }
        let num_shrinks = ((MAX_SIZE / size).max(1) * ROUNDS as usize) as u128;
        let mut total_ns = 0u128;
        for _ in 0..num_shrinks {
            let mut map = base.clone();
            let start = Instant::now();
            black_box(map.shrink_to(size));
            total_ns += start.elapsed().as_nanos();
        }
        total_ns as f64 / num_shrinks as f64
    }};
}

// ---------------------------------------------------------------------------
// TUI
// ---------------------------------------------------------------------------

#[cfg(feature = "tui")]
const IMPL_COLORS: [(&str, Color); 4] = [
    ("pomap", Color::Green),
    ("std_hashmap", Color::Yellow),
    ("hashbrown", Color::Red),
    ("pomap3", Color::Cyan),
];

#[cfg(feature = "tui")]
fn draw_tui(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
    chart_data: &HashMap<&str, Vec<(f64, f64)>>,
    title: &str,
    y_label: &str,
    status: &str,
) {
    let _ = terminal.draw(|f| {
        let chunks = RataLayout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(4), Constraint::Length(2)])
            .split(f.area());

        let x_labels: Vec<Span> = ["10", "100", "1K", "10K", "100K", "1M", "10M"]
            .iter()
            .map(|s| Span::raw(*s))
            .collect();

        let datasets: Vec<Dataset> = IMPL_COLORS
            .iter()
            .filter_map(|(impl_name, color)| {
                let data = chart_data.get(*impl_name)?;
                let leaked = Vec::leak(data.clone());
                Some(
                    Dataset::default()
                        .marker(symbols::Marker::Braille)
                        .graph_type(if cfg!(feature = "dots") {
                            ratatui::widgets::GraphType::Scatter
                        } else {
                            ratatui::widgets::GraphType::Line
                        })
                        .style(Style::default().fg(*color))
                        .data(leaked),
                )
            })
            .collect();

        let all_ys = || {
            IMPL_COLORS
                .iter()
                .filter_map(|(name, _)| chart_data.get(*name))
                .flat_map(|v| v.iter().map(|&(_, y)| y))
        };
        let max_y = all_ys().fold(0.0f64, f64::max);
        let min_y = all_ys().fold(f64::MAX, f64::min);
        let margin = (max_y - min_y).max(0.01) * 0.1;
        let y_lo = (min_y - margin).max(0.0);
        let y_hi = max_y + margin;

        let y_lo_s = format!("{:.0}ns", y_lo);
        let y_mid_s = format!("{:.0}ns", (y_lo + y_hi) / 2.0);
        let y_hi_s = format!("{:.0}ns", y_hi);

        let chart = Chart::new(datasets)
            .block(Block::default().title(title).borders(Borders::ALL))
            .x_axis(
                Axis::default()
                    .title("map_size")
                    .bounds([1.0, 7.0])
                    .labels(x_labels),
            )
            .y_axis(
                Axis::default()
                    .title(y_label)
                    .bounds([y_lo, y_hi])
                    .labels(vec![
                        Span::raw(y_lo_s),
                        Span::raw(y_mid_s),
                        Span::raw(y_hi_s),
                    ]),
            );

        f.render_widget(chart, chunks[0]);

        let bottom = RataLayout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Length(1)])
            .split(chunks[1]);

        let legend = Line::from(vec![
            Span::styled("■ pomap", Style::default().fg(Color::Green)),
            Span::raw("  "),
            Span::styled("■ std_hashmap", Style::default().fg(Color::Yellow)),
            Span::raw("  "),
            Span::styled("■ hashbrown", Style::default().fg(Color::Red)),
            Span::raw("  "),
            Span::styled("■ pomap3", Style::default().fg(Color::Cyan)),
        ]);
        f.render_widget(Paragraph::new(legend), bottom[0]);
        f.render_widget(Paragraph::new(status), bottom[1]);
    });
}

// ---------------------------------------------------------------------------
// Main benchmark runner
// ---------------------------------------------------------------------------

fn run_bench(name: &str, y_label: &str, mut measure: impl FnMut(usize, u8) -> f64) {
    let sizes = target_sizes();

    let csv_name = format!("{}_graph.csv", name);
    let mut file =
        std::fs::File::create(&csv_name).unwrap_or_else(|_| panic!("failed to create {}", csv_name));
    writeln!(file, "map_size,impl,per_op_ns").unwrap();

    let mut chart_data: HashMap<&str, Vec<(f64, f64)>> = HashMap::new();

    // --- TUI setup ---
    #[cfg(feature = "tui")]
    let original_hook = std::panic::take_hook();
    #[cfg(feature = "tui")]
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stderr(), LeaveAlternateScreen);
        original_hook(info);
    }));
    #[cfg(feature = "tui")]
    enable_raw_mode().unwrap();
    #[cfg(feature = "tui")]
    execute!(io::stderr(), EnterAlternateScreen).unwrap();
    #[cfg(feature = "tui")]
    let backend = CrosstermBackend::new(io::stderr());
    #[cfg(feature = "tui")]
    let mut terminal = Terminal::new(backend).unwrap();

    #[cfg(feature = "tui")]
    let quit = Arc::new(AtomicBool::new(false));
    #[cfg(feature = "tui")]
    let quit_bg = quit.clone();
    #[cfg(feature = "tui")]
    let _input_thread = std::thread::spawn(move || loop {
        if event::poll(Duration::from_millis(50)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if key.code == KeyCode::Char('q')
                    || (key.code == KeyCode::Char('c')
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL))
                {
                    quit_bg.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }
        if quit_bg.load(Ordering::Relaxed) {
            break;
        }
    });

    let bench_start = Instant::now();
    let total_steps = sizes.len() * 4;
    let mut completed = 0usize;
    let mut order_rng = StdRng::seed_from_u64(0x105E27);

    for &size in &sizes {
        let mut order = [0u8, 1, 2, 3];
        order.shuffle(&mut order_rng);

        for &idx in &order {
            #[cfg(feature = "tui")]
            if quit.load(Ordering::Relaxed) {
                break;
            }

            let impl_name = IMPL_NAMES[idx as usize];

            let eta = if completed > 0 {
                let elapsed = bench_start.elapsed().as_secs_f64();
                let remaining = (total_steps - completed) as f64 * elapsed / completed as f64;
                if remaining >= 3600.0 {
                    format!("{:.1}h", remaining / 3600.0)
                } else if remaining >= 60.0 {
                    format!("{:.0}m", remaining / 60.0)
                } else {
                    format!("{:.0}s", remaining)
                }
            } else {
                "...".to_string()
            };

            let status = format!(
                "size={:<10} | {:12} | {}/{} | ETA {}",
                size, impl_name, completed + 1, total_steps, eta
            );

            #[cfg(feature = "tui")]
            draw_tui(
                &mut terminal,
                &chart_data,
                name,
                y_label,
                &format!(" {} | 'q' or Ctrl+C to stop", status),
            );

            #[cfg(not(feature = "tui"))]
            eprint!("\r{}", status);

            let per_op_ns = measure(size, idx);

            writeln!(file, "{},{},{:.2}", size, impl_name, per_op_ns).unwrap();
            let x = (size as f64).log10();
            chart_data
                .entry(impl_name)
                .or_insert_with(Vec::new)
                .push((x, per_op_ns));

            completed += 1;
        }
    }

    #[cfg(not(feature = "tui"))]
    eprintln!("\rDone! ({} steps)                              ", completed);

    #[cfg(feature = "tui")]
    {
        if !quit.load(Ordering::Relaxed) {
            draw_tui(
                &mut terminal,
                &chart_data,
                name,
                y_label,
                " Done! Press any key to exit.",
            );
            while !quit.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        disable_raw_mode().unwrap();
        execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let bench = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "insert_allocate".to_string());

    match bench.as_str() {
        "insert_allocate" => {
            let keys = random_items(0xA11CE, MAX_SIZE);
            let values = random_items(0xFACE, MAX_SIZE);
            run_bench("insert_allocate", "ns/insert", |size, idx| match idx {
                0 => time_insert_alloc!(size, keys, values, pm!()),
                1 => time_insert_alloc!(size, keys, values, std!()),
                2 => time_insert_alloc!(size, keys, values, hb!()),
                _ => time_insert_alloc!(size, keys, values, pm3!()),
            });
        }

        "insert_preallocated" => {
            let keys = random_items(0xA11CE, MAX_SIZE);
            let values = random_items(0xFACE, MAX_SIZE);
            run_bench("insert_preallocated", "ns/insert", |size, idx| match idx {
                0 => time_insert_prealloc!(size, keys, values, pm!(size)),
                1 => time_insert_prealloc!(size, keys, values, std!(size)),
                2 => time_insert_prealloc!(size, keys, values, hb!(size)),
                _ => time_insert_prealloc!(size, keys, values, pm3!(size)),
            });
        }

        "get_hits" => {
            let keys = random_items(0xFEED, MAX_SIZE);
            let values = random_items(0x1CEBEEF, MAX_SIZE);
            run_bench("get_hits", "ns/get", |size, idx| match idx {
                0 => time_gets!(size, keys, values, keys, size, 0xC01DBEEF, pm!(size)),
                1 => time_gets!(size, keys, values, keys, size, 0xC01DBEEF, std!(size)),
                2 => time_gets!(size, keys, values, keys, size, 0xC01DBEEF, hb!(size)),
                _ => time_gets!(size, keys, values, keys, size, 0xC01DBEEF, pm3!(size)),
            });
        }

        "get_misses" => {
            let present_keys = random_items(0xABA1, MAX_SIZE);
            let present_values = random_items(0xCAB, MAX_SIZE);
            let mut present_set: HashSet<BenchKey, BenchHasherBuilder> =
                HashSet::with_capacity_and_hasher(
                    present_keys.len(),
                    BenchHasherBuilder::default(),
                );
            for k in &present_keys {
                present_set.insert(k.clone());
            }
            let miss_keys = generate_miss_keys(0xBA5E, MAX_SIZE, &present_set);
            drop(present_set);
            run_bench("get_misses", "ns/get", |size, idx| match idx {
                0 => time_gets!(size, present_keys, present_values, miss_keys, size, 0xC0FFEE42, pm!(size)),
                1 => time_gets!(size, present_keys, present_values, miss_keys, size, 0xC0FFEE42, std!(size)),
                2 => time_gets!(size, present_keys, present_values, miss_keys, size, 0xC0FFEE42, hb!(size)),
                _ => time_gets!(size, present_keys, present_values, miss_keys, size, 0xC0FFEE42, pm3!(size)),
            });
        }

        "update_existing" => {
            let keys = random_items(0xC0FFEE, MAX_SIZE);
            let initial_values = random_items(0xABC, MAX_SIZE);
            let update_values = random_items(0xDEF, MAX_SIZE);
            run_bench("update_existing", "ns/update", |size, idx| match idx {
                0 => time_updates!(size, keys, initial_values, update_values, pm!(size)),
                1 => time_updates!(size, keys, initial_values, update_values, std!(size)),
                2 => time_updates!(size, keys, initial_values, update_values, hb!(size)),
                _ => time_updates!(size, keys, initial_values, update_values, pm3!(size)),
            });
        }

        "get_hotset" => {
            let mut rng = StdRng::seed_from_u64(0xDEC0DE);
            let hot_keys: Vec<BenchKey> =
                (0..HOT_SET_SIZE).map(|_| random_bench_item(&mut rng)).collect();
            let mut all_keys: Vec<BenchKey> = hot_keys.clone();
            all_keys.extend(random_items(0xD00D, MAX_SIZE - HOT_SET_SIZE));
            let all_values = random_items(0xBADD, MAX_SIZE);
            run_bench("get_hotset", "ns/get", |size, idx| {
                let hot_count = HOT_SET_SIZE.min(size);
                match idx {
                    0 => time_gets!(size, all_keys, all_values, hot_keys, hot_count, 0xDEC0DE42, pm!(size)),
                    1 => time_gets!(size, all_keys, all_values, hot_keys, hot_count, 0xDEC0DE42, std!(size)),
                    2 => time_gets!(size, all_keys, all_values, hot_keys, hot_count, 0xDEC0DE42, hb!(size)),
                    _ => time_gets!(size, all_keys, all_values, hot_keys, hot_count, 0xDEC0DE42, pm3!(size)),
                }
            });
        }

        "remove_hits" => {
            let keys = random_items(0xD15EA5E, MAX_SIZE);
            let values = random_items(0xBEEFC0DE, MAX_SIZE);
            run_bench("remove_hits", "ns/remove", |size, idx| match idx {
                0 => time_remove_hits!(size, keys, values, pm!(size)),
                1 => time_remove_hits!(size, keys, values, std!(size)),
                2 => time_remove_hits!(size, keys, values, hb!(size)),
                _ => time_remove_hits!(size, keys, values, pm3!(size)),
            });
        }

        "remove_misses" => {
            let present_keys = random_items(0xC0FFEE, MAX_SIZE);
            let present_values = random_items(0xBADF00D, MAX_SIZE);
            let mut present_set: HashSet<BenchKey, BenchHasherBuilder> =
                HashSet::with_capacity_and_hasher(
                    present_keys.len(),
                    BenchHasherBuilder::default(),
                );
            for k in &present_keys {
                present_set.insert(k.clone());
            }
            let miss_keys = generate_miss_keys(0xF00DFACE, MAX_SIZE, &present_set);
            drop(present_set);
            run_bench("remove_misses", "ns/remove", |size, idx| match idx {
                0 => time_remove_misses!(size, present_keys, present_values, miss_keys, 0xBA5EBA11, pm!(size)),
                1 => time_remove_misses!(size, present_keys, present_values, miss_keys, 0xBA5EBA11, std!(size)),
                2 => time_remove_misses!(size, present_keys, present_values, miss_keys, 0xBA5EBA11, hb!(size)),
                _ => time_remove_misses!(size, present_keys, present_values, miss_keys, 0xBA5EBA11, pm3!(size)),
            });
        }

        "shrink_to" => {
            let keys = random_items(0x5A11CE, MAX_SIZE);
            let values = random_items(0xC0FFEE55, MAX_SIZE);
            run_bench("shrink_to", "ns/shrink", |size, idx| match idx {
                0 => time_shrink!(size, keys, values, pm!(size * SHRINK_OVER_ALLOC)),
                1 => time_shrink!(size, keys, values, std!(size * SHRINK_OVER_ALLOC)),
                2 => time_shrink!(size, keys, values, hb!(size * SHRINK_OVER_ALLOC)),
                _ => time_shrink!(size, keys, values, pm3!(size * SHRINK_OVER_ALLOC)),
            });
        }

        _ => {
            eprintln!("Unknown benchmark: {}", bench);
            eprintln!("Available: insert_allocate, insert_preallocated, get_hits, get_misses,");
            eprintln!("           update_existing, get_hotset, remove_hits, remove_misses, shrink_to");
            std::process::exit(1);
        }
    }
}
