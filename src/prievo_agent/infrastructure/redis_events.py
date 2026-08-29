import json
import logging


logger = logging.getLogger("prievo.redis")


class NullEventBus:
    def publish(self, run_id, event):
        return False

    def cache_status(self, run):
        return False

    def wait(self, run_id, timeout=1):
        return False


class RedisEventBus:
    """Redis 只加速事件通知和 hot status；所有事实仍从数据库读取。"""

    def __init__(self, redis_url):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Full Mode 需要安装 redis") from exc
        self.client = redis.Redis.from_url(redis_url,decode_responses=True,
                                           socket_timeout=2,socket_connect_timeout=2)

    def publish(self,run_id,event):
        try:
            payload={"sequence":event.sequence,"event_type":event.event_type,
                     "message":event.message,"payload":event.payload}
            self.client.publish("prievo:run:{}:events".format(run_id),
                                json.dumps(payload,ensure_ascii=False))
            return True
        except Exception as exc:
            logger.warning("Redis 事件通知失败，客户端将回退数据库轮询：%s",exc)
            return False

    def cache_status(self,run):
        try:
            self.client.setex("prievo:run:{}:status".format(run.id),60,
                              json.dumps({"status":run.status.value,"generation":run.generation}))
            return True
        except Exception as exc:
            logger.warning("Redis hot status 写入失败：%s",exc); return False

    def wait(self,run_id,timeout=1):
        try:
            pubsub=self.client.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe("prievo:run:{}:events".format(run_id))
            message=pubsub.get_message(timeout=timeout)
            pubsub.close()
            return bool(message)
        except Exception as exc:
            logger.warning("Redis SSE 通知不可用，继续轮询 MySQL：%s",exc); return False


class PublishingStore:
    def __init__(self,store,event_bus):
        self.store=store; self.event_bus=event_bus

    def __getattr__(self,name):
        return getattr(self.store,name)

    def append_event(self,run_id,event_type,message,**payload):
        event=self.store.append_event(run_id,event_type,message,**payload)
        self.event_bus.publish(run_id,event)
        return event

    def create_task_run(self,task,run,event_type,message,payload):
        event=self.store.create_task_run(task,run,event_type,message,payload)
        self.event_bus.publish(run.id,event)
        self.event_bus.cache_status(run)
        return event

    def save_run(self,run):
        result=self.store.save_run(run)
        self.event_bus.cache_status(self.store.get_run(run.id))
        return result

    def start_run(self,*args,**kwargs):
        run=self.store.start_run(*args,**kwargs)
        self.event_bus.cache_status(run)
        return run

    def pause_run(self,*args,**kwargs):
        run=self.store.pause_run(*args,**kwargs)
        self.event_bus.cache_status(run)
        return run

    def complete_run(self,*args,**kwargs):
        run=self.store.complete_run(*args,**kwargs)
        self.event_bus.cache_status(run)
        return run

    def fail_run(self,*args,**kwargs):
        run=self.store.fail_run(*args,**kwargs)
        self.event_bus.cache_status(run)
        return run

    def update_run_progress(self,*args,**kwargs):
        run=self.store.update_run_progress(*args,**kwargs)
        self.event_bus.cache_status(run)
        return run

    def close(self):
        self.store.close()
