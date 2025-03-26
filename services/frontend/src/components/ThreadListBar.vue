<script setup lang="ts">
import { ref } from "vue";
import { RouterLink } from "vue-router";

import ThreadTile from "./ThreadTile.vue";
import { Thread } from "../api/threads";

const threads = ref([] as Thread[]);
const sideBarOpened = ref(false);

async function query_threads(): Promise<void> {
  const t = await Thread.query();
  // threads.value = t.map((x) => x.valueOf());
  threads.value = t;
}

query_threads();
</script>

<template>
  <div class="viewport d-flex overflow-hidden vh-100" :class="{ 'show-side-bar': sideBarOpened }">
    <div class="sidebar bg-dark p-2 h-100 z-3">
      <div class="overflow-y-auto p-1 w-100">
        <RouterLink to="/">
          <img src="@/assets/logo.svg" alt="logo" class="d-block mx-auto mb-3" style="width: 100px;">
        </RouterLink>
        <ThreadTile v-for="t in threads" :thread="t" :key="t.id.toString()"></ThreadTile>
      </div>
    </div>
    <div class="main-area bg-dark bg-gradient h-100 z-0">
      <div class="appbar bg-black p-1 w-100">
        <button type="button" class="btn d-block d-md-none position-relative start-0 top-50 translate-middle-y" @click="sideBarOpened = !sideBarOpened">
          <span class="d-block material-icons-outlined text-white">menu</span>
        </button>
      </div>
      <div class="content w-100" @click="sideBarOpened = false">
        <slot>
          <div class="position-relative start-50 w-100 top-50 translate-middle">
            <img src="/mikucry.png" alt="mikucry" class="mw-100 opacity-50 p-1 position-relative start-50 translate-middle-x" style="mix-blend-mode: luminosity;">
            <span class="d-block position-relative text-center text-white">Where am I?</span>
          </div>
        </slot>
      </div>
    </div>
  </div>
</template>

<style lang="css" scoped>
.viewport {
  --sidebar-width: 300px;
  position: relative;
  left: calc(-1 * var(--sidebar-width));
  width: calc(100vw + var(--sidebar-width));
  transition: left 0.3s ease-in-out, width 0.3s ease-in-out;
}

.viewport.show-side-bar {
  left: 0;
  width: 100vw;
}

.sidebar {
  display: grid;
  grid-template-rows: 1fr auto;
  width: var(--sidebar-width);
}

.main-area {
  --appbar-height: 60px;
  width: calc(100% - var(--sidebar-width));
}

.appbar {
  height: var(--appbar-height);
}

.content {
  height: calc(100% - var(--appbar-height));
}

@media (min-width: 768px) {
  .viewport {
    left: 0;
    width: 100vw;
  }
}
</style>
