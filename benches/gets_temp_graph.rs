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
// Bench types & helpers (mirrored from pomap_bench.rs)
// ---------------------------------------------------------------------------

#[cfg(feature = "bench-string")]
type BenchType = String;
#[cfg(not(feature = "bench-string"))]
type BenchType = u64;

type BenchKey = BenchType;
type BenchValue = BenchType;

type BenchHasher = AHasher;
type BenchHasherBuilder = BuildHasherDefault<BenchHasher>;
type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchHashMap = HashMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchHashbrownMap = HashbrownMap<BenchKey, BenchValue, BenchHasherBuilder>;

fn std_hashmap_with_capacity(capacity: usize) -> BenchHashMap {
    BenchHashMap::with_capacity_and_hasher(capacity, BenchHasherBuilder::default())
}

fn hashbrown_with_capacity(capacity: usize) -> BenchHashbrownMap {
    BenchHashbrownMap::with_capacity_and_hasher(capacity, BenchHasherBuilder::default())
}

const MAX_SIZE: usize = 1_000_000;
const HOT_SET: usize = MAX_SIZE.isqrt();

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

fn build_pomap_maps(
    sizes: &[usize],
    keys: &[BenchKey],
    values: &[BenchValue],
) -> Vec<(usize, BenchPoMap)> {
    sizes
        .iter()
        .map(|&size| {
            let mut map: BenchPoMap =
                BenchPoMap::with_capacity_and_hasher(size, BenchHasherBuilder::default());
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            (size, map)
        })
        .collect()
}

fn build_std_maps(
    sizes: &[usize],
    keys: &[BenchKey],
    values: &[BenchValue],
) -> Vec<(usize, BenchHashMap)> {
    sizes
        .iter()
        .map(|&size| {
            let mut map = std_hashmap_with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            (size, map)
        })
        .collect()
}

fn build_hashbrown_maps(
    sizes: &[usize],
    keys: &[BenchKey],
    values: &[BenchValue],
) -> Vec<(usize, BenchHashbrownMap)> {
    sizes
        .iter()
        .map(|&size| {
            let mut map = hashbrown_with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            (size, map)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// TUI drawing
// ---------------------------------------------------------------------------

#[cfg(feature = "tui")]
const BENCH_NAMES: [&str; 3] = ["get_hits", "get_misses", "get_hotset"];
#[cfg(feature = "tui")]
const IMPL_COLORS: [(&str, Color); 3] = [
    ("pomap", Color::Green),
    ("std_hashmap", Color::Yellow),
    ("hashbrown", Color::Red),
];

#[cfg(feature = "tui")]
fn draw_tui(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
    chart_data: &HashMap<(&str, &str), Vec<(f64, f64)>>,
    status: &str,
) {
    let _ = terminal.draw(|f| {
        let chunks = RataLayout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Ratio(1, 4),
                Constraint::Ratio(1, 4),
                Constraint::Ratio(1, 4),
                Constraint::Min(2),
            ])
            .split(f.area());

        let x_labels: Vec<Span> = ["1", "10", "100", "1K", "10K", "100K"]
            .iter()
            .map(|s| Span::raw(*s))
            .collect();

        for (i, bench) in BENCH_NAMES.iter().enumerate() {
            let all_impl_data: Vec<(&str, &Vec<(f64, f64)>)> = IMPL_COLORS
                .iter()
                .filter_map(|(name, _)| {
                    chart_data
                        .get(&(*bench as &str, *name as &str))
                        .map(|d| (*name, d))
                })
                .collect();

            let avg_at = |x: f64| -> f64 {
                let (sum, count) = all_impl_data
                    .iter()
                    .filter_map(|(_, d)| d.iter().find(|(dx, _)| *dx == x).map(|(_, y)| *y))
                    .fold((0.0, 0usize), |(s, c), y| (s + y, c + 1));
                if count > 0 { sum / count as f64 } else { 1.0 }
            };

            let datasets: Vec<Dataset> = IMPL_COLORS
                .iter()
                .filter_map(|(impl_name, color)| {
                    let data = chart_data.get(&(*bench as &str, *impl_name as &str))?;
                    let ratio: Vec<(f64, f64)> =
                        data.iter().map(|&(x, y)| (x, y / avg_at(x))).collect();
                    let ratio = Vec::leak(ratio);
                    Some(
                        Dataset::default()
                            .marker(symbols::Marker::Braille)
                            .graph_type(if cfg!(feature = "dots") {
                                ratatui::widgets::GraphType::Scatter
                            } else {
                                ratatui::widgets::GraphType::Line
                            })
                            .style(Style::default().fg(*color))
                            .data(ratio),
                    )
                })
                .collect();

            let all_ratios = || {
                IMPL_COLORS
                    .iter()
                    .filter_map(|(name, _)| chart_data.get(&(*bench as &str, *name as &str)))
                    .flat_map(|v| v.iter().map(|&(x, y)| y / avg_at(x)))
            };
            let max_r = all_ratios().fold(1.0f64, f64::max);
            let min_r = all_ratios().fold(f64::MAX, f64::min).min(1.0);
            let margin = (max_r - min_r).max(0.01) * 0.15;
            let y_lo = (min_r - margin).max(0.0);
            let y_hi = max_r + margin;

            let y_lo_s = format!("{:.2}x", y_lo);
            let y_mid_s = format!("{:.2}x", (y_lo + y_hi) / 2.0);
            let y_hi_s = format!("{:.2}x", y_hi);

            let chart = Chart::new(datasets)
                .block(Block::default().title(*bench).borders(Borders::ALL))
                .x_axis(
                    Axis::default()
                        .title("gets_per_round")
                        .bounds([0.0, 5.0])
                        .labels(x_labels.clone()),
                )
                .y_axis(
                    Axis::default()
                        .title("vs avg")
                        .bounds([y_lo, y_hi])
                        .labels(vec![
                            Span::raw(y_lo_s),
                            Span::raw(y_mid_s),
                            Span::raw(y_hi_s),
                        ]),
                );

            f.render_widget(chart, chunks[i]);
        }

        let bottom = RataLayout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Length(1)])
            .split(chunks[3]);

        let legend = Line::from(vec![
            Span::styled("■ pomap", Style::default().fg(Color::Green)),
            Span::raw("  "),
            Span::styled("■ std_hashmap", Style::default().fg(Color::Yellow)),
            Span::raw("  "),
            Span::styled("■ hashbrown", Style::default().fg(Color::Red)),
        ]);
        f.render_widget(Paragraph::new(legend), bottom[0]);
        f.render_widget(Paragraph::new(status), bottom[1]);
    });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // GPR values: 1..10, then 2x steps up to 100K
    let mut gpr_values: Vec<usize> = (1..=10).collect();
    let mut v = 10.0f64;
    loop {
        v = (v * 2.0).ceil();
        let n = v as usize;
        if n > 100_000 {
            break;
        }
        if n != *gpr_values.last().unwrap() {
            gpr_values.push(n);
        }
    }
    if *gpr_values.last().unwrap() != 100_000 {
        gpr_values.push(100_000);
    }

    let target_sizes: Vec<usize> = vec![10, 100, 1_000, 10_000, 100_000, 1_000_000];
    let max_size = *target_sizes.last().unwrap();

    let keys: Vec<BenchKey> = random_items(0xFEED, max_size);
    let values: Vec<BenchValue> = random_items(0x1CEBEEF, max_size);

    let mut present_set: HashSet<BenchKey, BenchHasherBuilder> =
        HashSet::with_capacity_and_hasher(keys.len(), BenchHasherBuilder::default());
    for key in &keys {
        present_set.insert(key.clone());
    }
    let mut miss_rng = StdRng::seed_from_u64(0xBA5E);
    let mut miss_keys: Vec<BenchKey> = Vec::with_capacity(max_size);
    while miss_keys.len() < max_size {
        let candidate = random_bench_item(&mut miss_rng);
        if !present_set.contains(&candidate) {
            miss_keys.push(candidate);
        }
    }
    drop(present_set);

    let mut hot_rng = StdRng::seed_from_u64(0xDEC0DE);
    let hot_keys: Vec<BenchKey> = (0..HOT_SET)
        .map(|_| random_bench_item(&mut hot_rng))
        .collect();
    let mut hot_map_keys: Vec<BenchKey> = hot_keys.clone();
    hot_map_keys.extend(random_items(0xD00D, max_size - HOT_SET));
    let hot_map_values: Vec<BenchValue> = random_items(0xBADD, max_size);

    const ROUNDS: u64 = 100;

    let mut file = std::fs::File::create("get_graph.csv").expect("failed to create get_graph.csv");
    writeln!(file, "gets_per_round,bench,impl,map_size,per_get_ns").unwrap();

    // --- TUI setup (when enabled) ---
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

    let mut chart_data: HashMap<(&str, &str), Vec<(f64, f64)>> = HashMap::new();

    // Timing macro: benchmark each size independently, then average.
    macro_rules! time_gets {
        ($file:expr, $chart:expr, $bench:expr, $impl_name:expr, $maps:expr,
         $gpr:expr, $lookup_keys:expr, $seed_base:expr, $key_range:expr) => {{
            let mut sum_per_get = 0.0f64;
            for &(size, ref map) in $maps.iter() {
                let range = $key_range(size);
                let mut total_ns = 0u128;
                for round in 0..ROUNDS {
                    let seed = $seed_base ^ size as u64;
                    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(round));
                    let start = Instant::now();
                    for _ in 0..$gpr {
                        let idx = rng.random_range(0..range);
                        black_box(map.get(&$lookup_keys[idx]));
                    }
                    total_ns += start.elapsed().as_nanos();
                }
                sum_per_get += total_ns as f64 / ($gpr as f64 * ROUNDS as f64);
            }
            let per_get = sum_per_get / $maps.len() as f64;
            writeln!(
                $file,
                "{},{},{},all,{:.2}",
                $gpr, $bench, $impl_name, per_get
            )
            .unwrap();
            let x = ($gpr as f64).log10().max(0.0);
            $chart
                .entry(($bench, $impl_name))
                .or_insert_with(Vec::new)
                .push((x, per_get));
        }};
    }

    #[cfg(feature = "tui")]
    let quit = Arc::new(AtomicBool::new(false));
    #[cfg(feature = "tui")]
    let quit_bg = quit.clone();
    #[cfg(feature = "tui")]
    let _input_thread = std::thread::spawn(move || {
        loop {
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
        }
    });

    let bench_start = Instant::now();
    let total_steps = gpr_values.len() * 9;
    let mut completed_steps = 0usize;

    macro_rules! step {
        ($gpr:expr, $bench:expr, $impl_name:expr, $($rest:tt)*) => {{
            let eta = if completed_steps > 0 {
                let elapsed = bench_start.elapsed().as_secs_f64();
                let per_step = elapsed / completed_steps as f64;
                let remaining = (total_steps - completed_steps) as f64 * per_step;
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
                "gpr={:<6} | {} / {} | ETA {}",
                $gpr, $bench, $impl_name, eta
            );
            #[cfg(feature = "tui")]
            draw_tui(&mut terminal, &chart_data, &format!(" {} | 'q' or Ctrl+C to stop", status));
            #[cfg(not(feature = "tui"))]
            eprint!("\r{}", status);
            time_gets!(file, chart_data, $bench, $impl_name, $($rest)*);
            completed_steps += 1;
        }};
    }

    macro_rules! check_quit_labeled {
        ($label:lifetime) => {
            #[cfg(feature = "tui")]
            if quit.load(Ordering::Relaxed) {
                break $label;
            }
        };
    }

    // Each bench type: build all 3 impls' maps, then for each GPR value run impls in random
    // order. Measure per-size independently. Drop all maps between bench types.

    let mut order_rng = StdRng::seed_from_u64(0x0DE12);

    // get_hits
    {
        let pm = build_pomap_maps(&target_sizes, &keys, &values);
        let sm = build_std_maps(&target_sizes, &keys, &values);
        let hb = build_hashbrown_maps(&target_sizes, &keys, &values);
        #[allow(unused_labels)]
        'hits: for (_, &gpr) in gpr_values.iter().enumerate() {
            let mut order = [0u8, 1, 2];
            order.shuffle(&mut order_rng);
            for &idx in &order {
                check_quit_labeled!('hits);
                match idx {
                    0 => step!(
                        gpr,
                        "get_hits",
                        "pomap",
                        pm,
                        gpr,
                        keys,
                        0xC01DBEEF,
                        |s: usize| s
                    ),
                    1 => step!(
                        gpr,
                        "get_hits",
                        "std_hashmap",
                        sm,
                        gpr,
                        keys,
                        0xC01DBEEF,
                        |s: usize| s
                    ),
                    _ => step!(
                        gpr,
                        "get_hits",
                        "hashbrown",
                        hb,
                        gpr,
                        keys,
                        0xC01DBEEF,
                        |s: usize| s
                    ),
                }
            }
        }
    }

    // get_misses — maps built from keys/values, lookups use miss_keys
    {
        let pm = build_pomap_maps(&target_sizes, &keys, &values);
        let sm = build_std_maps(&target_sizes, &keys, &values);
        let hb = build_hashbrown_maps(&target_sizes, &keys, &values);
        #[allow(unused_labels)]
        'misses: for (_, &gpr) in gpr_values.iter().enumerate() {
            let mut order = [0u8, 1, 2];
            order.shuffle(&mut order_rng);
            for &idx in &order {
                check_quit_labeled!('misses);
                match idx {
                    0 => step!(
                        gpr,
                        "get_misses",
                        "pomap",
                        pm,
                        gpr,
                        miss_keys,
                        0xC0FFEE42,
                        |s: usize| s
                    ),
                    1 => step!(
                        gpr,
                        "get_misses",
                        "std_hashmap",
                        sm,
                        gpr,
                        miss_keys,
                        0xC0FFEE42,
                        |s: usize| s
                    ),
                    _ => step!(
                        gpr,
                        "get_misses",
                        "hashbrown",
                        hb,
                        gpr,
                        miss_keys,
                        0xC0FFEE42,
                        |s: usize| s
                    ),
                }
            }
        }
    }

    // get_hotset
    {
        let pm = build_pomap_maps(&target_sizes, &hot_map_keys, &hot_map_values);
        let sm = build_std_maps(&target_sizes, &hot_map_keys, &hot_map_values);
        let hb = build_hashbrown_maps(&target_sizes, &hot_map_keys, &hot_map_values);
        #[allow(unused_labels)]
        'hotset: for (_, &gpr) in gpr_values.iter().enumerate() {
            let mut order = [0u8, 1, 2];
            order.shuffle(&mut order_rng);
            for &idx in &order {
                check_quit_labeled!('hotset);
                match idx {
                    0 => step!(
                        gpr,
                        "get_hotset",
                        "pomap",
                        pm,
                        gpr,
                        hot_keys,
                        0xDEC0DE42,
                        |s: usize| HOT_SET.min(s).max(1)
                    ),
                    1 => step!(
                        gpr,
                        "get_hotset",
                        "std_hashmap",
                        sm,
                        gpr,
                        hot_keys,
                        0xDEC0DE42,
                        |s: usize| HOT_SET.min(s).max(1)
                    ),
                    _ => step!(
                        gpr,
                        "get_hotset",
                        "hashbrown",
                        hb,
                        gpr,
                        hot_keys,
                        0xDEC0DE42,
                        |s: usize| HOT_SET.min(s).max(1)
                    ),
                }
            }
        }
    }

    #[cfg(not(feature = "tui"))]
    eprintln!(
        "\rDone! ({} steps)                              ",
        completed_steps
    );

    #[cfg(feature = "tui")]
    {
        if !quit.load(Ordering::Relaxed) {
            draw_tui(&mut terminal, &chart_data, " Done! Press any key to exit.");
            while !quit.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        disable_raw_mode().unwrap();
        execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
    }
}
