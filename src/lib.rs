#![deny(clippy::all)]
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::JsNumber;
use std::sync::{Arc, Mutex};

// https://github.com/rust-lang/rust-analyzer/issues/17429
use napi_derive::napi;

#[napi(js_name = "FSRS")]
#[derive(Debug)]
pub struct FSRS(Arc<Mutex<fsrs::FSRS>>);

#[napi]
/// directly use fsrs::DEFAULT_PARAMETERS will cause error.
/// referencing statics in constants is unstable
/// see issue #119618 <https://github.com/rust-lang/rust/issues/119618> for more information
/// `static` and `const` variables can refer to other `const` variables. A `const` variable, however, cannot refer to a `static` variable.
/// to fix this, the value can be extracted to a `const` and then used.
pub const DEFAULT_PARAMETERS: [f32; 19] = [
  0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604, 0.0046, 1.54575, 0.1192, 1.01925,
  1.9395, 0.11, 0.29605, 2.2698, 0.2315, 2.9898, 0.51655, 0.6621,
];
impl Default for FSRS {
  fn default() -> Self {
    Self::new(None)
  }
}

#[derive(Debug)]
pub struct ProgressData {
  pub current: usize,
  pub total: usize,
  pub percent: f64,
}

/// A background task that runs `compute_parameters`, sending progress updates via TSFN.
pub struct ComputeParametersTask {
  // Thread-safe reference to your FSRS model
  model: Arc<Mutex<fsrs::FSRS>>,
  // Training data, made owned so it doesn't reference `&self`
  train_data: Vec<fsrs::FSRSItem>,
  // The threadsafe JS callback for partial updates
  progress_callback: ThreadsafeFunction<ProgressData, ErrorStrategy::CalleeHandled>,
}

impl Task for ComputeParametersTask {
  type Output = Vec<f32>;
  type JsValue = Vec<f64>;

  fn compute(&mut self) -> Result<Self::Output> {
    // 1) Create a shared progress object
    let progress_state = fsrs::CombinedProgressState::new_shared();
    let progress_state_for_thread = Arc::clone(&progress_state);
    // Clone what we need for the separate thread
    let train_data = self.train_data.clone();
    let model = Arc::clone(&self.model);

    // 2) Spawn a new thread that does the heavy lifting
    //    so we can poll progress *in parallel* on this thread.
    let handle = std::thread::spawn(move || -> Result<Vec<f32>> {
      let locked = model.lock().map_err(|_| {
        Error::new(
          Status::GenericFailure,
          "Failed to lock FSRS model".to_string(),
        )
      })?;

      // Now use `progress_state_for_thread` inside the closure
      locked
        .compute_parameters(train_data, Some(progress_state_for_thread), true)
        .map_err(|e| Error::new(Status::GenericFailure, format!("{:?}", e)))
    });

    // 3) Meanwhile, on *this* thread, poll `progress_state` in a loop
    //    and call `progress_callback` with updated progress.
    loop {
      let (current, total, finished) = {
        let p = progress_state.lock().unwrap();
        (p.current(), p.total(), p.finished())
      };

      let percent = if total == 0 {
        0.0
      } else {
        (current as f64 / total as f64) as f64
      };

      // Call JS callback if you want once per second or whenever progress changes
      let status = self.progress_callback.call(
        Ok(ProgressData {
          current,
          total,
          percent,
        }),
        ThreadsafeFunctionCallMode::NonBlocking,
      );

      if status != napi::Status::Ok {
        eprintln!("Failed to call JS callback, status = {:?}", status);
      }

      if finished || percent >= 100.0 {
        break;
      }

      // Sleep briefly before polling again
      std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // 4) Join the compute thread to get the final result
    let final_result = handle.join().map_err(|_| {
      Error::new(
        Status::GenericFailure,
        "Panic occurred in compute thread".to_string(),
      )
    })??; // `??` to unwrap the `Result` from inside the thread

    // 5) Return the final result
    Ok(final_result)
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output.iter().map(|&x| x as f64).collect())
  }
}

#[napi]
impl FSRS {
  #[napi(constructor)]
  pub fn new(parameters: Option<Vec<JsNumber>>) -> Self {
    let params: [f32; 19] = match parameters {
      Some(parameters) => {
        let mut array = [0.0; 19];
        for (i, value) in parameters.iter().enumerate().take(19) {
          array[i] = value.get_double().unwrap_or(0.0) as f32;
        }
        array
      }
      None => DEFAULT_PARAMETERS,
    };
    Self(Arc::new(Mutex::new(
      fsrs::FSRS::new(Some(&params)).unwrap(),
    )))
  }

  #[napi(ts_return_type = "Promise<number[]>")]
  pub fn compute_parameters(
    &self,
    train_set: Vec<&FSRSItem>,
    #[napi(
      ts_arg_type = "(err: null | Error, value: { current: number, total: number, percent: number }) => void"
    )]
    progress_js_fn: JsFunction,
    signal: Option<AbortSignal>,
  ) -> Result<AsyncTask<ComputeParametersTask>> {
    // Convert your `JS` training items to owned `fsrs::FSRSItem`
    let train_data = train_set
      .into_iter()
      .map(|item| item.0.clone())
      .collect::<Vec<_>>();

    // Turn `JsFunction` into a `ThreadsafeFunction`
    let tsfn: ThreadsafeFunction<ProgressData, ErrorStrategy::CalleeHandled> = progress_js_fn
      .create_threadsafe_function(0, |ctx| {
        let progress_data: ProgressData = ctx.value;
        let env = ctx.env;
        let mut progress_obj = env.create_object()?;

        progress_obj.set("current", env.create_uint32(progress_data.current as u32)?)?;
        progress_obj.set("total", env.create_uint32(progress_data.total as u32)?)?;
        progress_obj.set("percent", progress_data.percent as f32)?;

        // Return it as the single argument to JS
        Ok(vec![progress_obj])
      })?;

    // We assume `FSRS` internally stores `Arc<Mutex<fsrs::FSRS>>` or something similar
    // so it's already thread-safe. If not, you must fix that part first.
    let task = ComputeParametersTask {
      model: Arc::clone(&self.0), // e.g., if `this.0` is `Arc<Mutex<fsrs::FSRS>>`
      train_data,
      progress_callback: tsfn,
    };

    let result = match signal {
      Some(signal) => AsyncTask::with_signal(task, signal),
      None => AsyncTask::new(task),
    };
    Ok(result)
  }

  #[napi]
  pub fn next_states(
    &self,
    current_memory_state: Option<&MemoryState>,
    desired_retention: f64,
    days_elapsed: u32,
  ) -> NextStates {
    let locked_model = self.0.lock().unwrap();
    NextStates(
      locked_model
        .next_states(
          current_memory_state.map(|x| x.0),
          desired_retention as f32,
          days_elapsed,
        )
        .unwrap(),
    )
  }

  #[napi]
  pub fn benchmark(&self, train_set: Vec<&FSRSItem>) -> Vec<f32> {
    let locked_model = self.0.lock().unwrap();
    locked_model.benchmark(train_set.iter().map(|x| x.0.clone()).collect(), true)
  }

  #[napi]
  pub fn memory_state_from_sm2(
    &self,
    ease_factor: f64,
    interval: f64,
    sm2_retention: f64,
  ) -> MemoryState {
    let locked_model = self.0.lock().unwrap();
    MemoryState(
      locked_model
        .memory_state_from_sm2(ease_factor as f32, interval as f32, sm2_retention as f32)
        .unwrap(),
    )
  }

  #[napi]
  pub fn memory_state(&self, item: &FSRSItem, starting_state: Option<&MemoryState>) -> MemoryState {
    let locked_model = self.0.lock().unwrap();
    MemoryState(
      locked_model
        .memory_state(item.0.clone(), starting_state.map(|x| x.0))
        .unwrap(),
    )
  }
}

#[napi(js_name = "FSRSReview")]
#[derive(Debug)]
pub struct FSRSReview(fsrs::FSRSReview);

#[napi]
impl FSRSReview {
  #[napi(constructor)]
  pub fn new(rating: u32, delta_t: u32) -> Self {
    Self(fsrs::FSRSReview { rating, delta_t })
  }
  #[napi(getter)]
  pub fn rating(&self) -> u32 {
    self.0.rating
  }
  #[napi(getter)]
  pub fn delta_t(&self) -> u32 {
    self.0.delta_t
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "FSRSItem")]
#[derive(Debug)]
pub struct FSRSItem(fsrs::FSRSItem);
#[napi]
impl FSRSItem {
  #[napi(constructor)]
  pub fn new(reviews: Vec<&FSRSReview>) -> Self {
    Self(fsrs::FSRSItem {
      reviews: reviews.iter().map(|x| x.0).collect(),
    })
  }

  #[napi(getter)]
  pub fn reviews(&self) -> Vec<FSRSReview> {
    self.0.reviews.iter().map(|x| FSRSReview(*x)).collect()
  }

  #[napi]
  pub fn long_term_review_cnt(&self) -> u32 {
    self.0.long_term_review_cnt() as u32
  }

  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "MemoryState")]
#[derive(Debug)]
pub struct MemoryState(fsrs::MemoryState);
#[napi]
impl MemoryState {
  #[napi(constructor)]
  pub fn new(stability: f64, difficulty: f64) -> Self {
    Self(fsrs::MemoryState {
      stability: stability as f32,
      difficulty: difficulty as f32,
    })
  }
  #[napi(getter)]
  pub fn stability(&self) -> f64 {
    self.0.stability as f64
  }
  #[napi(getter)]
  pub fn difficulty(&self) -> f64 {
    self.0.difficulty as f64
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "NextStates")]
#[derive(Debug)]
pub struct NextStates(fsrs::NextStates);
#[napi]
impl NextStates {
  #[napi(getter)]
  pub fn hard(&self) -> ItemState {
    ItemState(self.0.hard.clone())
  }
  #[napi(getter)]
  pub fn good(&self) -> ItemState {
    ItemState(self.0.good.clone())
  }
  #[napi(getter)]
  pub fn easy(&self) -> ItemState {
    ItemState(self.0.easy.clone())
  }
  #[napi(getter)]
  pub fn again(&self) -> ItemState {
    ItemState(self.0.again.clone())
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}

#[napi(js_name = "ItemState")]
#[derive(Debug)]
pub struct ItemState(fsrs::ItemState);
#[napi]
impl ItemState {
  #[napi(getter)]
  pub fn memory(&self) -> MemoryState {
    MemoryState(self.0.memory)
  }
  #[napi(getter)]
  pub fn interval(&self) -> f32 {
    self.0.interval
  }
  #[napi(js_name = "toJSON")]
  pub fn to_json(&self) -> String {
    format!("{:?}", self.0)
  }
}
