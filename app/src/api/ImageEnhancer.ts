import type {
  EnhancementResult,
  TaskStatus,
  WorkerInputMessage,
  WorkerOutputMessage,
} from '../types/task';


interface ResultWaiter {
  resolve: (result: EnhancementResult) => void;
  reject: (error: Error) => void;
}

interface TaskRecord {
  status: TaskStatus;
  file: File;
  result: EnhancementResult | null;
  waiters: ResultWaiter[];
}


const DEFAULT_PROCESSING_TIMEOUT_MS = 30_000;


export class ImageEnhancer extends EventTarget {
  private readonly tasks =
    new Map<string, TaskRecord>();

  private readonly queue: string[] = [];

  private worker: Worker;

  private activeTaskId: string | null = null;

  private activeTimeoutId:
    ReturnType<typeof setTimeout> | null = null;

  private readonly processingTimeoutMs: number;


  constructor(
    processingTimeoutMs =
      DEFAULT_PROCESSING_TIMEOUT_MS,
  ) {
    super();

    if (
      !Number.isFinite(processingTimeoutMs) ||
      processingTimeoutMs <= 0
    ) {
      throw new RangeError(
        'Лимит времени должен быть положительным числом.',
      );
    }

    this.processingTimeoutMs =
      processingTimeoutMs;

    this.worker = this.createWorker();
  }


  async submit(file: File): Promise<string> {
    if (!(file instanceof File)) {
      throw new TypeError(
        'На обработку необходимо передать файл.',
      );
    }

    if (file.size === 0) {
      throw new Error(
        'Выбранный файл пуст.',
      );
    }

    const taskId = crypto.randomUUID();

    const initialStatus: TaskStatus = {
      taskId,
      status: 'queued',
      progress: 0,
      error: null,
    };

    this.tasks.set(
      taskId,
      {
        status: initialStatus,
        file,
        result: null,
        waiters: [],
      },
    );

    this.queue.push(taskId);

    this.emitStatus(initialStatus);
    this.startNextTask();

    return taskId;
  }


  async getStatus(
    taskId: string,
  ): Promise<TaskStatus> {
    const task = this.getTask(taskId);

    return {
      ...task.status,
    };
  }


  async getResult(
    taskId: string,
  ): Promise<EnhancementResult> {
    const task = this.getTask(taskId);

    if (task.result) {
      return task.result;
    }

    if (task.status.status === 'failed') {
      throw new Error(
        task.status.error ??
          'Задача завершилась с ошибкой.',
      );
    }

    if (task.status.status === 'cancelled') {
      throw new Error(
        'Задача была отменена.',
      );
    }

    return new Promise<EnhancementResult>(
      (resolve, reject) => {
        task.waiters.push(
          {
            resolve,
            reject,
          },
        );
      },
    );
  }


  async cancel(
    taskId: string,
  ): Promise<boolean> {
    const task = this.getTask(taskId);

    if (
      this.isTerminalStatus(
        task.status.status,
      )
    ) {
      return false;
    }

    if (this.activeTaskId === taskId) {
      this.restartWorker();
    } else {
      const queueIndex =
        this.queue.indexOf(taskId);

      if (queueIndex !== -1) {
        this.queue.splice(
          queueIndex,
          1,
        );
      }
    }

    task.status = {
      taskId,
      status: 'cancelled',
      progress: task.status.progress,
      error: null,
    };

    for (const waiter of task.waiters) {
      waiter.reject(
        new Error(
          'Задача была отменена.',
        ),
      );
    }

    task.waiters = [];

    this.emitStatus(task.status);
    this.startNextTask();

    return true;
  }


  release(taskId: string): boolean {
    const task = this.tasks.get(taskId);

    if (
      !task ||
      !this.isTerminalStatus(
        task.status.status,
      )
    ) {
      return false;
    }

    task.waiters.length = 0;

    return this.tasks.delete(taskId);
  }


  dispose(): void {
    this.clearActiveTimeout();
    this.worker.terminate();

    for (const task of this.tasks.values()) {
      if (
        !this.isTerminalStatus(
          task.status.status,
        )
      ) {
        for (
          const waiter of task.waiters
        ) {
          waiter.reject(
            new Error(
              'Модуль обработки остановлен.',
            ),
          );
        }
      }
    }

    this.queue.length = 0;
    this.activeTaskId = null;
    this.tasks.clear();
  }


  private createWorker(): Worker {
    const worker = new Worker(
      new URL(
        '../workers/enhancement.worker.ts',
        import.meta.url,
      ),
      {
        type: 'module',
      },
    );

    worker.onmessage = (
      event: MessageEvent<WorkerOutputMessage>,
    ): void => {
      this.handleWorkerMessage(
        event.data,
      );
    };

    worker.onerror = (
      event: ErrorEvent,
    ): void => {
      this.handleWorkerError(
        event.message ||
          'Ошибка Web Worker.',
      );
    };

    return worker;
  }


  private startNextTask(): void {
    if (this.activeTaskId) {
      return;
    }

    while (this.queue.length > 0) {
      const taskId =
        this.queue.shift();

      if (!taskId) {
        return;
      }

      const task =
        this.tasks.get(taskId);

      if (
        !task ||
        this.isTerminalStatus(
          task.status.status,
        )
      ) {
        continue;
      }

      this.activeTaskId = taskId;

      const message: WorkerInputMessage = {
        type: 'process',
        taskId,
        file: task.file,
      };

      this.activeTimeoutId = setTimeout(
        () => {
          this.handleTaskTimeout(
            taskId,
          );
        },
        this.processingTimeoutMs,
      );

      try {
        this.worker.postMessage(
          message,
        );
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : 'Не удалось запустить обработку.';

        this.restartWorker();

        this.failTask(
          taskId,
          errorMessage,
        );
      }

      return;
    }
  }


  private handleWorkerMessage(
    message: WorkerOutputMessage,
  ): void {
    const task =
      this.tasks.get(message.taskId);

    if (
      !task ||
      this.isTerminalStatus(
        task.status.status,
      )
    ) {
      return;
    }

    if (message.type === 'status') {
      task.status = {
        taskId: message.taskId,
        status: message.status,
        progress: message.progress,
        error: null,
      };

      this.emitStatus(task.status);

      return;
    }

    if (message.type === 'result') {
      if (
        this.activeTaskId ===
        message.taskId
      ) {
        this.clearActiveTimeout();
        this.activeTaskId = null;
      }

      task.result = message.result;

      task.status = {
        taskId: message.taskId,
        status: 'completed',
        progress: 100,
        error: null,
      };

      for (
        const waiter of task.waiters
      ) {
        waiter.resolve(
          message.result,
        );
      }

      task.waiters = [];

      this.emitStatus(task.status);
      this.startNextTask();

      return;
    }

    if (message.type === 'error') {
      this.failTask(
        message.taskId,
        message.error,
      );
    }
  }


  private handleWorkerError(
    errorMessage: string,
  ): void {
    const failedTaskId =
      this.activeTaskId;

    this.restartWorker();

    if (failedTaskId) {
      this.failTask(
        failedTaskId,
        errorMessage,
      );
    } else {
      this.startNextTask();
    }
  }


  private failTask(
    taskId: string,
    errorMessage: string,
  ): void {
    const task =
      this.tasks.get(taskId);

    if (!task) {
      return;
    }

    if (
      this.isTerminalStatus(
        task.status.status,
      )
    ) {
      return;
    }

    if (
      this.activeTaskId === taskId
    ) {
      this.clearActiveTimeout();
      this.activeTaskId = null;
    }

    task.status = {
      taskId,
      status: 'failed',
      progress: task.status.progress,
      error: errorMessage,
    };

    for (
      const waiter of task.waiters
    ) {
      waiter.reject(
        new Error(errorMessage),
      );
    }

    task.waiters = [];

    this.emitStatus(task.status);
    this.startNextTask();
  }


  private handleTaskTimeout(
    taskId: string,
  ): void {
    if (
      this.activeTaskId !== taskId
    ) {
      return;
    }

    const timeoutSeconds =
      Math.ceil(
        this.processingTimeoutMs /
          1000,
      );

    this.restartWorker();

    this.failTask(
      taskId,
      `Обработка превысила допустимое время (${timeoutSeconds} с).`,
    );
  }


  private clearActiveTimeout(): void {
    if (
      this.activeTimeoutId === null
    ) {
      return;
    }

    clearTimeout(
      this.activeTimeoutId,
    );

    this.activeTimeoutId = null;
  }


  private restartWorker(): void {
    this.clearActiveTimeout();

    this.worker.terminate();
    this.worker = this.createWorker();

    this.activeTaskId = null;
  }


  private getTask(
    taskId: string,
  ): TaskRecord {
    const task =
      this.tasks.get(taskId);

    if (!task) {
      throw new Error(
        `Задача ${taskId} не найдена.`,
      );
    }

    return task;
  }


  private isTerminalStatus(
    status: TaskStatus['status'],
  ): boolean {
    return (
      status === 'completed' ||
      status === 'cancelled' ||
      status === 'failed'
    );
  }


  private emitStatus(
    status: TaskStatus,
  ): void {
    this.dispatchEvent(
      new CustomEvent<TaskStatus>(
        'statuschange',
        {
          detail: {
            ...status,
          },
        },
      ),
    );
  }
}