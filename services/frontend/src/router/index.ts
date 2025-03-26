import { createRouter, createWebHistory } from "vue-router";

import HomeView from "../views/HomeView.vue";
import ThreadView from "../views/ThreadView.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      component: HomeView,
    },
    {
      path: "/threads",
      component: HomeView,
    },
    {
      path: "/threads/:id",
      component: ThreadView,
    },
  ],
});

export default router;
