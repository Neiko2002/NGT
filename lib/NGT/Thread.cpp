//
// Copyright (C) 2015 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include	<mutex>
#include	<thread>
#include	<optional>
#include	<condition_variable>

#include	"Thread.h"

using namespace std;
using namespace NGT;

namespace NGT {
class ThreadInfo {
  public:
    std::optional<std::thread> thread;
};

class ThreadMutex {
  public:
    std::recursive_mutex mutex;
    std::condition_variable_any condition;
};
}

Thread::Thread() {
  threadInfo = new ThreadInfo;
  threadNo = -1;
  isTerminate = false;
}

Thread::~Thread() {
  if (threadInfo != 0) {
    delete threadInfo;
  }
}

ThreadMutex *
Thread::constructThreadMutex()
{
  return new ThreadMutex;
}

void
Thread::destructThreadMutex(ThreadMutex *t)
{
  if (t != 0) {
    delete t;
  }
}

int
Thread::start()
{
  threadInfo->thread.emplace(Thread::startThread, this);
  return 0;
}

int
Thread::join()
{
  if (threadInfo->thread && threadInfo->thread->joinable()) {
    threadInfo->thread->join();
  }
  return 0;
}

void
Thread::lock(ThreadMutex &m)
{
  m.mutex.lock();
}
void
Thread::unlock(ThreadMutex &m)
{
  m.mutex.unlock();
}
void
Thread::signal(ThreadMutex &m)
{
  m.condition.notify_one();
}

void
Thread::wait(ThreadMutex &m)
{
  std::unique_lock lock{m.mutex};
  m.condition.wait(lock);
}

void
Thread::broadcast(ThreadMutex &m)
{
  m.condition.notify_all();
}

void
Thread::mutexInit(ThreadMutex &m)
{
}
