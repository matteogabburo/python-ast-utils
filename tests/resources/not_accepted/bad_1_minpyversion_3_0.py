#!/usr/bin/env python
# encoding:utf-8
import json
from app import db, redis
from app.utils.times import Time
from importlib import import_module
from copy import deepcopy


def add_execute_logs(socket, uuid, app_uuid, app_name, result):
    log_info = {
        "uuid": uuid,
        "app_uuid": app_uuid,
        "app_name": app_name,
        "result": result,
        "create_time": Time.get_date_time(),
    }

    db.table("zbn_logs").insert(log_info)

    data = {"method": "execute_log", "data": log_info}

    if socket is None:
        pass
    else:
        socket.send(json.dumps(data))


def params_attrs(attrs_str, global_data):
    item = deepcopy(global_data)
    for attrs in attrs_str.split("."):
        if type(item) is dict:
            if attrs in item.keys():
                item = item[attrs]
            else:
                return False, "错误:不存在的key:{}".format(attrs)
        elif type(item) is list:
            try:
                index = int(attrs)
            except:
                return False, "key错误:{},对象为array,无法获取对应值".format(attrs)
            else:
                if len(item) >= index:
                    item = item[index]
                else:
                    return False, "宏错误:{},超出结果最大条数".format(attrs)
    return True, item


def render_args(kwargs: dict, global_data: dict):
    kw = deepcopy(kwargs)
    for k, v in kwargs.items():
        if "{{" in v and "}}" in v:
            s = v.index("{{")
            e = v.index("}}") + 2
            keys = v[s:e]
            key = keys.replace("{{", "").replace("}}", "").strip()
            s, value = params_attrs(key, global_data)
            if s:
                if type(value) is list:
                    result = []
                    for i in value:
                        kw[k] = v.replace(keys, i)
                        result.append(deepcopy(kw))
                    return True, result
                else:
                    kw[k] = v.replace(keys, value)
                    return True, [kw]
            else:
                return False, value
    return True, [kw]


def execute(
    app_dir, global_data, data=None,
):
    import_path = "app.core.apps." + str(app_dir) + ".main"
    mode = import_module(import_path)
    func_name = data["action"]
    try:
        func = getattr(mode, func_name)
    except AttributeError:
        return False, {}, "action错误:未找到对应action:{}".format(func_name)

    kwargs = {
        k: v
        for k, v in data.items()
        if k != "node_name" and k != "action" and k != "app"
    }

    s, kwargs = render_args(kwargs, global_data)

    if not s:
        return False, kwargs
    try:
        s, result = func(**kwargs)
    except Exception as e:
        return False, "Action Error: {}".format(e)

    return s, result

    # args = ""
    # for key in data:
    #     if key != "node_name" and key != "action" and key != "app":
    #         args = args + "," + key

    # eval_action = "app.{action}({args})".format(action=data["action"], args=args[1:])
    # result = eval(eval_action)

    # return result


def get_app_data(socket, uuid, app_uuid, app_info, global_data):
    key = app_uuid + "_result"
    if redis.exists(key) == 0:
        s, result = execute(
            app_dir=app_info["app_dir"], data=app_info["data"], global_data=global_data
        )
        if not s:
            return s, result
        global_data.update(result["data"])
        output = result["output"]
        redis.set(key, output, ex=3)

        print("uuid : ", app_uuid)
        print("name : ", app_info["data"]["node_name"])
        print("result : ", output)
        print("===================================")

        add_execute_logs(
            socket=socket,
            uuid=uuid,
            app_uuid=app_uuid,
            app_name=app_info["data"]["node_name"],
            result=output,
        )

        return s, result
    else:
        return True, redis.get(key).decode()


def find_start_app(edges, start_app=None):
    for r in edges:
        if start_app:
            if str(r["source"]) == start_app:
                return r["target"]


def find_next_app(edges, next_app=None):
    num = 0
    for r in edges:
        if next_app:
            key = next_app + "_sum"
            if redis.exists(key) == 1:
                sum = redis.get(key)

                if str(r["source"]) == next_app:
                    if num != int(sum):
                        num = num + 1
                    else:
                        return r["label"], r["source"], r["target"]
            else:
                if str(r["source"]) == next_app:
                    return r["label"], r["source"], r["target"]


def find_next_apps(edges, next_app=None):
    num = 0
    for r in edges:
        if next_app:
            key = next_app + "_sum"
            if redis.exists(key) == 1:
                sum = redis.get(key)

                if str(r["source"]) == next_app:
                    if num != int(sum):
                        num = num + 1
                    else:
                        return r["label"], r["source"], r["target"]
            else:
                if str(r["source"]) == next_app:
                    return r["label"], r["source"], r["target"]


def run_exec(socket, uuid):
    workflow_info = (
        db.table("zbn_workflow")
        .select("uuid", "name", "start_app", "end_app", "flow_json", "flow_data")
        .where("uuid", uuid)
        .first()
    )

    if workflow_info:
        start_app = workflow_info["start_app"]
        end_app = workflow_info["end_app"]

        flow_json = json.loads(workflow_info["flow_json"])
        flow_data = json.loads(workflow_info["flow_data"])

        # for r in flow_json["edges"]:
        #     print(r["label"], r["source"], r["target"])

        global_data = {}

        target_app = find_start_app(edges=flow_json["edges"], start_app=start_app)

        add_execute_logs(
            socket=socket, uuid=uuid, app_uuid=start_app, app_name="开始", result="剧本开始执行"
        )

        is_while = True

        while is_while:
            try:
                # 拿到当前APP数据
                if_else, source_app, next_app = find_next_app(
                    edges=flow_json["edges"], next_app=target_app
                )
            except Exception:
                add_execute_logs(
                    socket=socket,
                    uuid=uuid,
                    app_uuid="",
                    app_name="",
                    result="当前剧本不具有可执行条件",
                )
                is_while = False
                break

            key = target_app + "_sum"
            if redis.exists(key) == 1:
                sum = redis.get(key)
                redis.set(key, int(sum) + 1, ex=3)
            else:
                redis.set(key, 1, ex=3)

            # 当前APP执行数据
            source_info = flow_data[source_app]
            # print(source_app)
            s, ifelse_result = get_app_data(
                socket=socket,
                uuid=uuid,
                app_uuid=source_app,
                app_info=source_info,
                global_data=global_data,
            )

            if not s:
                add_execute_logs(
                    socket=socket,
                    uuid=uuid,
                    app_uuid=end_app,
                    app_name=flow_data.get(source_app).get("name"),
                    result="执行错误:{}".format(ifelse_result),
                )
                add_execute_logs(
                    socket=socket,
                    uuid=uuid,
                    app_uuid=end_app,
                    app_name="结束",
                    result="剧本执行结束",
                )
                is_while = False

            if if_else != "":
                if if_else == ifelse_result:
                    target_app = next_app
            else:
                target_app = next_app

            if next_app == end_app:
                add_execute_logs(
                    socket=socket,
                    uuid=uuid,
                    app_uuid=end_app,
                    app_name="结束",
                    result="剧本执行结束",
                )
                is_while = False


def run_app_demo(data):
    app_dir = data.get("app_dir")
    action_data = data.get("data")
    import_path = "app.core.apps." + str(app_dir) + ".main"
    mode = import_module(import_path)
    func_name = action_data.get("action")
    try:
        func = getattr(mode, func_name)
    except AttributeError:
        return False, {}, "action错误:未找到对应action:{}".format(func_name)
    kwargs = {
        k: v
        for k, v in action_data.items()
        if k != "node_name" and k != "action" and k != "app"
    }
    kw = kwargs.values()
    try:
        s, result = func(*kw)
    except Exception as e:
        return False, "Action Error: {}".format(e)
    return s, result


class WorkFlowException(Exception):
    def __init__(self, error):
        self.error_msg = error
        print(error)


class WorkFlow:
    envs = {}

    def __init__(self, uuid, flow_json, flow_data, socket):
        self.uuid = uuid
        self.flow_json = json.loads(flow_json)
        self.flow_data = json.loads(flow_data)
        self.start_node = self.__start_app()
        self.end_node = self.__end_app()
        self.socket = socket

    def __start_app(self):
        for n in self.flow_json.get("nodes"):
            if n.get("label") == "开始":
                self.start_node = n
                return n
        raise WorkFlowException("流程中未找到<开始>节点")

    def __end_app(self):
        for n in self.flow_json.get("nodes"):
            if n.get("label") == "结束":
                self.end_node = n
                return n
        raise WorkFlowException("流程中未找到<结束>节点")

    def __get_flow_node(self):
        result = {self.start_node["id"]: []}
        source_id_set = [self.start_node["id"]]

        while True:
            item = []
            for edge in self.flow_json.get("edges"):
                source_id = edge["source"]
                target_id = edge["target"]
                if source_id in source_id_set:
                    item.append(target_id)
                    if source_id in result.keys():
                        result[source_id].append(target_id)
                    else:
                        result[source_id] = [
                            target_id,
                        ]
                    if target_id not in result.keys():
                        result[target_id] = []
            source_id_set = item
            if len(item) == 0:
                break
        if self.end_node["id"] not in result.keys():
            raise WorkFlowException("流程没有结束节点")
        self.flow_nodes = result
        return result

    def check_app_data(self):
        flow_node = self.__get_flow_node()
        for k, v in flow_node.items():
            if k in [self.start_node["id"], self.end_node["id"]]:
                continue
            node_data = self.flow_data.get(k)
            if node_data.get("data", None) is None:
                raise WorkFlowException("节点<{}>没有配置参数")
        return self.flow_nodes

    def run(self):
        try:
            self.check_app_data()
        except WorkFlowException as e:
            add_execute_logs(
                socket=self.socket,
                uuid=self.uuid,
                app_uuid="",
                app_name="",
                result="剧本错误:{}".format(e),
            )
            self.end_of_flow()
            return

        source_apps = self.flow_nodes.get(self.start_node["id"])
        add_execute_logs(
            socket=self.socket,
            uuid=self.uuid,
            app_uuid=self.start_node["id"],
            app_name="开始",
            result="剧本开始执行",
        )

        is_while = True
        while is_while:
            next_apps = []
            for app_id in source_apps:
                app_data = self.flow_data.get(app_id, None)

                if not app_data:
                    """节点数据不存在，一般是结束节点"""
                    continue

                app_dir = app_data["app_dir"]
                data = app_data["data"]
                action = data["action"]

                # 渲染参数数据
                print("data", data)
                s, kwargs = self.__render_kwargs(data)

                if not s:
                    add_execute_logs(
                        socket=self.socket,
                        uuid=self.uuid,
                        app_uuid=app_id,
                        app_name=app_data.get("name"),
                        result="参数错误:{}".format(kwargs),
                    )
                    self.end_of_flow()
                    return
                for kw in kwargs:
                    s, result = self.__run_action(app_dir, kw, action)

                    if not s:
                        """运行异常"""
                        add_execute_logs(
                            socket=self.socket,
                            uuid=self.uuid,
                            app_uuid=app_id,
                            app_name=app_data.get("name"),
                            result="执行错误:{}".format(result),
                        )
                        self.end_of_flow()
                        return

                    else:
                        """返回结果，更新变量"""
                        self.envs.update(result["data"])
                        output = result["output"]
                        add_execute_logs(
                            socket=self.socket,
                            uuid=self.uuid,
                            app_uuid=app_id,
                            app_name=app_data["name"],
                            result=output,
                        )

                    next_apps.extend(self.flow_nodes.get(app_id))
            source_apps = next_apps
            if len(source_apps) == 0:
                is_while = False
        self.end_of_flow()

    def __render_kwargs(self, data):
        kwargs = {
            k: v
            for k, v in data.items()
            if k != "node_name" and k != "action" and k != "app"
        }
        s, kwargs = render_args(kwargs, self.envs)
        if not s:
            return False, kwargs
        else:
            result = [i.values() for i in kwargs]
            return True, result

    def __run_action(self, app_dir, kw, action):
        import_path = "app.core.apps." + str(app_dir) + ".main"
        mode = import_module(import_path)
        func_name = action
        try:
            func = getattr(mode, func_name)
        except AttributeError:
            return False, {}, "action错误:未找到对应action:{}".format(func_name)

        try:
            s, result = func(*kw)
        except Exception as e:
            return False, "Action Error: {}".format(e)

        return s, result

    def end_of_flow(self):
        add_execute_logs(
            socket=self.socket,
            uuid=self.uuid,
            app_uuid=self.end_node["id"],
            app_name="结束",
            result="剧本执行结束",
        )
