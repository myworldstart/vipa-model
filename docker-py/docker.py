# This is a sample Python script.
import docker
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def test():
    client = docker.DockerClient(base_url='tcp://172.19.186.248:2375')
    container = client.containers.run(image='anibali/pytorch:latest',
                                      command='/bin/bash',
                                      user='root',
                                      name='gpu_test1',
                                      volumes=['/home/test:/home/test'],
                                      working_dir='/home/liyanpeng',
                                      tty=True,
                                      detach=True,
                                      stdin_open=True,
                                      environment=['PYTHONPATH=xxxxx:$PYTHONPATH'],
                                      device_requests=[
                                          docker.types.DeviceRequest(device_ids=['0'], capabilities=[['gpu']])]
                                      )

    result = container.exec_run('python /home/test/t.py')
    print(result.exit_code)
    # container = client.containers.get('d64cfefcd3d2')
    # container_id = container.id
    # result = container.exec_run('python /home/test/t.py')
    # print(result.exit_code)
    # exec_command = 'python /home/test/t.py'
    # print(1);
    # exec_create_response = client.api.exec_create(container_id, exec_command)
    # print(1);
    # exec_start_response = client.api.exec_start(exec_create_response['Id'])
    # print(1);
    # print(exec_start_response.decode('utf-8'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
