import type {
  TaskStatus,
  WorkerInputMessage,
  WorkerOutputMessage,
} from '../types/task';

interface ResultWaiter {
  resolve: (blob: Blob) => void;
  reject: (error: Error) => void;
}

interface TaskRecord {
  status: TaskStatus;
  worker: Worker;
  result: Blob | null;
  waiters: ResultWaiter[];
}

export class ImageEnhancer extends EventTarget {
  private readonly tasks = new Map<string, TaskRecord>();

  async submit(file: File): Promise<string> {
    if (!(file instanceof File)) {
      throw new TypeError('На обработку необходимо передать файл.');
    }

    if (file.size === 0) {
      throw new Error('Выбранный файл пуст.');
    }

    const taskId = crypto.randomUUID();

    const worker = new Worker(
      new URL('../workers/enhancement.worker.ts', import.meta.url),
      {
        type: 'module',
      },
    );

    const initialStatus: TaskStatus = {
      taskId,
      status: 'queued',
      progress: 0,
      error: null,
    };

    const task: TaskRecord = {
      status: initialStatus,
      worker,
      result: null,
      waiters: [],
    };

    this.tasks.set(taskId, task);
    this.emitStatus(initialStatus);

    worker.onmessage = (
      event: MessageEvent<WorkerOutputMessage>,
    ): void => {
      this.handleWorkerMessage(taskId, event.data);
    };

    worker.onerror = (event: ErrorEvent): void => {
      this.failTask(taskId, event.message || 'Ошибка Web Worker.');
    };

    const message: WorkerInputMessage = {
      type: 'process',
      taskId,
      file,
    };

    worker.postMessage(message);

    return taskId;
  }

  async getStatus(taskId: string): Promise<TaskStatus> {
    const task = this.getTask(taskId);

    return { ...task.status };
  }

  async getResult(taskId: string): Promise<Blob> {
    const task = this.getTask(taskId);

    if (task.result) {
      return task.result;
    }

    if (task.status.status === 'failed') {
      throw new Error(task.status.error ?? 'Задача завершилась с ошибкой.');
    }

    if (task.status.status === 'cancelled') {
      throw new Error('Задача была отменена.');
    }

    return new Promise<Blob>((resolve, reject) => {
      task.waiters.push({ resolve, reject });
    });
  }

  async cancel(taskId: string): Promise<boolean> {
    const task = this.getTask(taskId);

    if (
      task.status.status === 'completed' ||
      task.status.status === 'failed' ||
      task.status.status === 'cancelled'
    ) {
      return false;
    }

    /*
     * terminate() сразу останавливает Worker.
     * Для MVP это наиболее простой способ отмены.
     */
    task.worker.terminate();

    task.status = {
      taskId,
      status: 'cancelled',
      progress: task.status.progress,
      error: null,
    };

    for (const waiter of task.waiters) {
      waiter.reject(new Error('Задача была отменена.'));
    }

    task.waiters = [];
    this.emitStatus(task.status);

    return true;
  }

  private handleWorkerMessage(
    taskId: string,
    message: WorkerOutputMessage,
  ): void {
    const task = this.tasks.get(taskId);

    if (!task) {
      return;
    }

    if (message.type === 'status') {
      task.status = {
        taskId,
        status: message.status,
        progress: message.progress,
        error: null,
      };

      this.emitStatus(task.status);
      return;
    }

    if (message.type === 'result') {
      task.result = message.blob;

      task.status = {
        taskId,
        status: 'completed',
        progress: 100,
        error: null,
      };

      for (const waiter of task.waiters) {
        waiter.resolve(message.blob);
      }

      task.waiters = [];
      task.worker.terminate();

      this.emitStatus(task.status);
      return;
    }

    if (message.type === 'error') {
      this.failTask(taskId, message.error);
    }
  }

  private failTask(taskId: string, errorMessage: string): void {
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

    task.worker.terminate();

    for (const waiter of task.waiters) {
      waiter.reject(new Error(errorMessage));
    }

    task.waiters = [];
    this.emitStatus(task.status);
  }

  private getTask(taskId: string): TaskRecord {
    const task = this.tasks.get(taskId);

    if (!task) {
      throw new Error(`Задача ${taskId} не найдена.`);
    }

    return task;
  }

  private emitStatus(status: TaskStatus): void {
    this.dispatchEvent(
      new CustomEvent<TaskStatus>('statuschange', {
        detail: { ...status },
      }),
    );
  }
}
