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

export class ImageEnhancer extends EventTarget {
  private readonly tasks = new Map<string, TaskRecord>();

  private readonly queue: string[] = [];

  private worker: Worker;

  private activeTaskId: string | null = null;

  constructor() {
    super();
    this.worker = this.createWorker();
  }

  async submit(file: File): Promise<string> {
    if (!(file instanceof File)) {
      throw new TypeError(
        'На обработку необходимо передать файл.',
      );
    }

    if (file.size === 0) {
      throw new Error('Выбранный файл пуст.');
    }

    const taskId = crypto.randomUUID();

    const initialStatus: TaskStatus = {
      taskId,
      status: 'queued',
      progress: 0,
      error: null,
    };

    this.tasks.set(taskId, {
      status: initialStatus,
      file,
      result: null,
      waiters: [],
    });

    this.queue.push(taskId);
    this.emitStatus(initialStatus);
    this.startNextTask();

    return taskId;
  }

  async getStatus(taskId: string): Promise<TaskStatus> {
    const task = this.getTask(taskId);

    return { ...task.status };
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
      throw new Error('Задача была отменена.');
    }

    return new Promise<EnhancementResult>(
      (resolve, reject) => {
        task.waiters.push({
          resolve,
          reject,
        });
      },
    );
  }

  async cancel(taskId: string): Promise<boolean> {
    const task = this.getTask(taskId);

    if (this.isTerminalStatus(task.status.status)) {
      return false;
    }

    if (this.activeTaskId === taskId) {
      this.worker.terminate();
      this.worker = this.createWorker();
      this.activeTaskId = null;
    } else {
      const queueIndex = this.queue.indexOf(taskId);

      if (queueIndex !== -1) {
        this.queue.splice(queueIndex, 1);
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
        new Error('Задача была отменена.'),
      );
    }

    task.waiters = [];

    this.emitStatus(task.status);
    this.startNextTask();

    return true;
  }

  dispose(): void {
    this.worker.terminate();

    for (const task of this.tasks.values()) {
      if (!this.isTerminalStatus(task.status.status)) {
        for (const waiter of task.waiters) {
          waiter.reject(
            new Error('Модуль обработки остановлен.'),
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
      this.handleWorkerMessage(event.data);
    };

    worker.onerror = (event: ErrorEvent): void => {
      this.handleWorkerError(
        event.message || 'Ошибка Web Worker.',
      );
    };

    return worker;
  }

  private startNextTask(): void {
    if (this.activeTaskId) {
      return;
    }

    while (this.queue.length > 0) {
      const taskId = this.queue.shift();

      if (!taskId) {
        return;
      }

      const task = this.tasks.get(taskId);

      if (
        !task ||
        this.isTerminalStatus(task.status.status)
      ) {
        continue;
      }

      this.activeTaskId = taskId;

      const message: WorkerInputMessage = {
        type: 'process',
        taskId,
        file: task.file,
      };

      this.worker.postMessage(message);

      return;
    }
  }

  private handleWorkerMessage(
    message: WorkerOutputMessage,
  ): void {
    const task = this.tasks.get(message.taskId);

    if (!task) {
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
      task.result = message.result;

      task.status = {
        taskId: message.taskId,
        status: 'completed',
        progress: 100,
        error: null,
      };

      for (const waiter of task.waiters) {
        waiter.resolve(message.result);
      }

      task.waiters = [];

      if (this.activeTaskId === message.taskId) {
        this.activeTaskId = null;
      }

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
    const failedTaskId = this.activeTaskId;

    this.worker.terminate();
    this.worker = this.createWorker();
    this.activeTaskId = null;

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
    const task = this.tasks.get(taskId);

    if (!task) {
      return;
    }

    task.status = {
      taskId,
      status: 'failed',
      progress: task.status.progress,
      error: errorMessage,
    };

    for (const waiter of task.waiters) {
      waiter.reject(
        new Error(errorMessage),
      );
    }

    task.waiters = [];

    if (this.activeTaskId === taskId) {
      this.activeTaskId = null;
    }

    this.emitStatus(task.status);
    this.startNextTask();
  }

  private getTask(taskId: string): TaskRecord {
    const task = this.tasks.get(taskId);

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

  private emitStatus(status: TaskStatus): void {
    this.dispatchEvent(
      new CustomEvent<TaskStatus>(
        'statuschange',
        {
          detail: { ...status },
        },
      ),
    );
  }
}